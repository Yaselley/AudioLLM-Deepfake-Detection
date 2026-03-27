"""
Evaluation script for Thinking-mode AudioLLM deepfake detection models.

Loads a trained checkpoint, runs inference on an eval dataset, and writes
a JSONL output with: path, label (ground-truth), response (model output).
Also prints accuracy and per-class metrics at the end.

Usage:
  python eval_thinking.py --config <train_config.yaml> --eval_data <data.jsonl> \
      [--output_dir <dir>] [--batch_size 4] [--max_new_tokens 1024]
"""

import os
import re
import json
import yaml
import argparse
import logging
import time
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from huggingface_hub import login
from peft import PeftModel

from models_clean.audiollm_base_thinking import AudioDeepfakeLLMThinking
from dataset_thinking import AudioDeepfakeThinkingDataset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_prediction(response_text):
    """Extract the final Real/Fake prediction from a thinking-mode response."""
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response_text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = response_text.strip()

    answer_lower = answer.lower()
    if "fake" in answer_lower:
        return "Fake"
    elif "real" in answer_lower:
        return "Real"
    return answer


def main(args):
    if args.hf_token:
        login(token=args.hf_token)
    else:
        login(token="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")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "./output")
    eval_output_dir = args.output_dir or os.path.join(output_dir, "eval_results")
    os.makedirs(eval_output_dir, exist_ok=True)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(eval_output_dir, "eval.log")),
            logging.StreamHandler()
        ]
    )

    seed = config["train"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Config: {args.config}")
    logging.info(f"Eval data: {args.eval_data}")
    logging.info(f"Checkpoint dir: {output_dir}")

    # ── Build Model ──
    model_cfg = config["model"]
    model = AudioDeepfakeLLMThinking(
        llm_path=model_cfg["llm_path"],
        llm_type=model_cfg.get("llm_type", "qwen2"),
        audio_encoder_path=model_cfg.get("audio_encoder_path", None),
        audio_encoder_type=model_cfg.get("audio_encoder_type", "whisper"),
        audio_encoder_source=model_cfg.get("audio_encoder_source", "huggingface"),
        freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True),
        use_weighted_layer_sum=model_cfg.get("use_weighted_layer_sum", False),
        weighted_layer_norm=model_cfg.get("weighted_layer_norm", "none"),
        lora=model_cfg.get("lora", True),
        lora_rank=model_cfg.get("lora_rank", 64),
        lora_alpha=model_cfg.get("lora_alpha", 16),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        special_tokens=model_cfg.get("special_tokens", []),
        device_8bit=0,
        speechtokenizer_num_layers=model_cfg.get("speechtokenizer_num_layers", 8),
        speechtokenizer_codebook_size=model_cfg.get("speechtokenizer_codebook_size", 1024),
        speechtokenizer_return_discrete=model_cfg.get("speechtokenizer_return_discrete", True),
        dac_num_codebooks=model_cfg.get("dac_num_codebooks", 12),
        dac_codebook_size=model_cfg.get("dac_codebook_size", 1024),
        dac_model_type=model_cfg.get("dac_model_type", "16khz"),
        dac_return_discrete=model_cfg.get("dac_return_discrete", True),
        sqcodec_num_quantizers=model_cfg.get("sqcodec_num_quantizers", 4),
        sqcodec_num_levels=model_cfg.get("sqcodec_num_levels", 4096),
        use_opensmile=model_cfg.get("use_opensmile", False),
        opensmile_only=model_cfg.get("opensmile_only", False),
    )

    # ── Load Checkpoint ──
    is_lora = model_cfg.get("lora", True)
    if is_lora:
        lora_path = os.path.join(output_dir, "lora_best")
        if not os.path.exists(lora_path):
            logging.error(f"LoRA checkpoint not found at {lora_path}")
            return
        logging.info(f"Loading LoRA weights from {lora_path}")
        model.llm_model = PeftModel.from_pretrained(
            model.llm_model.base_model.model, lora_path,
            is_trainable=False
        )

        proj_path = os.path.join(output_dir, "proj_best.pt")
        if os.path.exists(proj_path) and model.speech_proj is not None:
            model.speech_proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
            logging.info(f"Loaded projection weights from {proj_path}")

        audio_enc_path = os.path.join(output_dir, "audio_encoder_best.pt")
        if os.path.exists(audio_enc_path) and model.audio_encoder is not None:
            model.audio_encoder.load_state_dict(torch.load(audio_enc_path, map_location="cpu"))
            logging.info(f"Loaded audio encoder weights from {audio_enc_path}")
    else:
        ckpt_path = os.path.join(output_dir, "checkpoint_best.pt")
        if not os.path.exists(ckpt_path):
            logging.error(f"Checkpoint not found at {ckpt_path}")
            return
        logging.info(f"Loading full checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    model.to(device)
    model.eval()

    # ── Eval Dataset ──
    data_cfg = config["data"]
    eval_dataset = AudioDeepfakeThinkingDataset(
        args.eval_data,
        prompt_path=data_cfg.get("prompt_path", None),
        system_prompt_path=data_cfg.get("system_prompt_path", None),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    generate_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
    }

    eval_name = os.path.splitext(os.path.basename(args.eval_data))[0]
    result_path = os.path.join(eval_output_dir, f"{eval_name}_results.jsonl")

    logging.info(f"Running evaluation on {len(eval_dataset)} samples...")
    logging.info(f"Results will be saved to {result_path}")

    total, correct = 0, 0
    tp, fp, tn, fn = 0, 0, 0, 0
    start_time = time.time()

    with open(result_path, 'w') as out_f:
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                raw_wav = batch["raw_wav"].to(device)
                audio_attention_mask = batch.get("audio_attention_mask", None)
                if audio_attention_mask is not None:
                    audio_attention_mask = audio_attention_mask.to(device)

                samples = {
                    "raw_wav": raw_wav,
                    "audio_attention_mask": audio_attention_mask,
                    "prompt": batch["prompt"],
                }

                responses = model.generate(samples, generate_cfg=generate_cfg)

                print(responses[0])

                paths = batch["path"]
                labels = batch["text"]
                print(labels[0])
                
                for i, response in enumerate(responses):
                    gt_label_raw = labels[i]
                    if "fake" in gt_label_raw.lower():
                        gt = "Fake"
                    else:
                        gt = "Real"

                    pred = parse_prediction(response)

                    is_correct = (pred.lower() == gt.lower())
                    if is_correct:
                        correct += 1
                    total += 1

                    if gt == "Fake" and pred == "Fake":
                        tp += 1
                    elif gt == "Real" and pred == "Fake":
                        fp += 1
                    elif gt == "Real" and pred == "Real":
                        tn += 1
                    elif gt == "Fake" and pred == "Real":
                        fn += 1

                    result = {
                        "path": paths[i],
                        "label": gt,
                        "prediction": pred,
                        "response": response,
                    }
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

                if (batch_idx + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    acc = correct / total * 100 if total > 0 else 0
                    logging.info(f"  Batch {batch_idx+1}/{len(eval_loader)} | "
                                 f"Acc so far: {acc:.2f}% ({correct}/{total}) | "
                                 f"Time: {elapsed:.1f}s")

    elapsed = time.time() - start_time

    # ── Metrics ──
    accuracy = correct / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    logging.info("=" * 60)
    logging.info(f"Evaluation Complete on: {args.eval_data}")
    logging.info(f"  Total samples: {total}")
    logging.info(f"  Accuracy:  {accuracy:.2f}%  ({correct}/{total})")
    logging.info(f"  Precision: {precision:.2f}%  (Fake detected correctly)")
    logging.info(f"  Recall:    {recall:.2f}%  (Fake samples caught)")
    logging.info(f"  F1 Score:  {f1:.2f}%")
    logging.info(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    logging.info(f"  Time: {elapsed:.1f}s")
    logging.info(f"  Results: {result_path}")
    logging.info("=" * 60)

    metrics = {
        "eval_data": args.eval_data,
        "config": args.config,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "elapsed_sec": elapsed,
    }
    metrics_path = os.path.join(eval_output_dir, f"{eval_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Thinking-mode AudioLLM")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML (same one used for training)")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to eval JSONL (e.g. data/itw.jsonl)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir for results (default: <train_output>/eval_results)")
    parser.add_argument("--batch_size", type=int, default=4, help="Eval batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token")
    args = parser.parse_args()
    main(args)
