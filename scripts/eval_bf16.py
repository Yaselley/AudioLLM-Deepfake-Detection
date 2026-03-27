import os
import argparse
import logging
import torch
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from contextlib import nullcontext
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, det_curve

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Make CuDNN deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} for reproducibility")

set_seed(42)

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_clean.audiollm_base_opensmile import AudioDeepfakeLLM
from dataset import AudioDeepfakeDataset

from huggingface_hub import login
login(token="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")


def load_model(checkpoint_dir, epoch, device, compile_model=False, use_amp=False, amp_dtype='fp16'):
    """Load model with optimizations"""
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found in {checkpoint_dir}")
    
    # Initialize Model
    model_cfg = config["model"]
    # model = AudioDeepfakeLLM(
    #     llm_path=model_cfg["llm_path"],
    #     llm_type=model_cfg.get("llm_type", "qwen2"),
    #     audio_encoder_path=model_cfg["audio_encoder_path"],
    #     audio_encoder_type=model_cfg.get("audio_encoder_type", "whisper"),
    #     audio_encoder_source=model_cfg.get("audio_encoder_source", "huggingface"),
    #     freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True),
    #     speechtokenizer_num_layers=model_cfg.get("speechtokenizer_num_layers", 8),
    #     speechtokenizer_codebook_size=model_cfg.get("speechtokenizer_codebook_size", 1024),
    #     lora=model_cfg.get("lora", True),
    #     lora_rank=model_cfg.get("lora_rank", 64),
    #     lora_alpha=model_cfg.get("lora_alpha", 16),
    #     lora_dropout=model_cfg.get("lora_dropout", 0.05),
    #     special_tokens=model_cfg.get("special_tokens", []),
    #     device_8bit=0 
    # )
    
    model = AudioDeepfakeLLM(
        llm_path=model_cfg["llm_path"],
        llm_type=model_cfg.get("llm_type", "qwen2"),
        audio_encoder_path=model_cfg.get("audio_encoder_path", None),
        audio_encoder_type=model_cfg.get("audio_encoder_type", "whisper"),
        audio_encoder_source=model_cfg.get("audio_encoder_source", "huggingface"),
        freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True),
        lora=model_cfg.get("lora", True),
        lora_rank=model_cfg.get("lora_rank", 64),
        lora_alpha=model_cfg.get("lora_alpha", 16),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        special_tokens=model_cfg.get("special_tokens", []),
        device_8bit=0,
        # SpeechTokenizer parameters
        speechtokenizer_num_layers=model_cfg.get("speechtokenizer_num_layers", 8),
        speechtokenizer_codebook_size=model_cfg.get("speechtokenizer_codebook_size", 1024),
        speechtokenizer_return_discrete=model_cfg.get("speechtokenizer_return_discrete", True),  # ADD THIS LINE
        dac_num_codebooks=model_cfg.get("dac_num_codebooks", 12),
        dac_codebook_size=model_cfg.get("dac_codebook_size", 1024),
        dac_model_type=model_cfg.get("dac_model_type", "16khz"),
        dac_return_discrete=model_cfg.get("dac_return_discrete", True),
        # SQ-Codec parameters
        sqcodec_num_quantizers=model_cfg.get("sqcodec_num_quantizers", 4),
        sqcodec_num_levels=model_cfg.get("sqcodec_num_levels", 4096),
        use_weighted_layer_sum = model_cfg.get("use_weighted_layer_sum", False),
        weighted_layer_norm = model_cfg.get("weighted_layer_norm", "none"),

        use_opensmile=model_cfg.get("use_opensmile", False),
        opensmile_only=model_cfg.get("opensmile_only", False),
    )
    
    # Load Projector Weights
    if model.speech_proj is not None:
        proj_path = os.path.join(checkpoint_dir, f"proj_{epoch}.pt")
        if os.path.exists(proj_path):
            print(f"Loading projector weights from {proj_path}")
            model.speech_proj.load_state_dict(torch.load(proj_path, map_location="cpu", weights_only=True))
    
    # # Load Audio Encoder Weights (required for use_weighted_layer_sum: layer_pooler weights)
    # audio_enc_path = os.path.join(checkpoint_dir, f"audio_encoder_{epoch}.pt")
    # if os.path.exists(audio_enc_path):
    #     print(f"Loading audio encoder weights from {audio_enc_path}")
    #     model.audio_encoder.load_state_dict(torch.load(audio_enc_path, map_location="cpu", weights_only=True))
    # elif model_cfg.get("use_weighted_layer_sum", False):
    #     print(
    #         "WARNING: use_weighted_layer_sum=True but audio_encoder_{} not found. "
    #         "Layer-pooler weights are untrained (init). Re-train with updated train_tokens.py to save them.".format(epoch)
    #     )

    # Load Audio Encoder Weights (required for use_weighted_layer_sum: layer_pooler weights)
    # audio_enc_path = os.path.join(checkpoint_dir, f"audio_encoder_{epoch}.pt")
    # if os.path.exists(audio_enc_path):
    #     print(f"Loading audio encoder weights from {audio_enc_path}")
    #     # Load raw dictionary
    #     audio_state_dict = torch.load(audio_enc_path, map_location="cpu", weights_only=True)
        
    #     # Create a new dictionary to fix the keys
    #     fixed_state_dict = {}
    #     for k, v in audio_state_dict.items():
    #         # The error shows model expects "encoder.conv1..." but file has "conv1..."
    #         # We explicitly add the prefix if it's missing
    #         if not k.startswith("encoder."):
    #             new_key = f"encoder.{k}"
    #         else:
    #             new_key = k
    #         fixed_state_dict[new_key] = v
            
    #     # Load the fixed dictionary
    #     # We use strict=False to avoid crashing on minor unused keys, but print the result
    #     load_result = model.audio_encoder.load_state_dict(fixed_state_dict, strict=False)
        
    #     print("Audio Encoder loaded.")
    #     if len(load_result.missing_keys) > 0:
    #         print(f"WARNING: Missing keys in audio_encoder: {load_result.missing_keys[:5]} ...")
    #     if len(load_result.unexpected_keys) > 0:
    #         print(f"WARNING: Unexpected keys in audio_encoder: {load_result.unexpected_keys[:5]} ...")

    # elif model_cfg.get("use_weighted_layer_sum", False):
    #     print(
    #         "WARNING: use_weighted_layer_sum=True but audio_encoder_{} not found. "
    #         "Layer-pooler weights are untrained (init). Re-train with updated train_tokens.py to save them.".format(epoch)
    #     )

    # Load Audio Encoder Weights
    audio_enc_path = os.path.join(checkpoint_dir, f"audio_encoder_{epoch}.pt")
    if os.path.exists(audio_enc_path):
        print(f"Loading audio encoder weights from {audio_enc_path}")
        state_dict = torch.load(audio_enc_path, map_location="cpu", weights_only=True)
        
        # --- FIX: Intelligent Key Matching ---
        # Get the actual keys your current model expects
        model_state_dict = model.audio_encoder.state_dict()
        expected_keys = set(model_state_dict.keys())
        fixed_state_dict = {}
        
        print("Adapting checkpoint keys to model structure...")
        
        for k, v in state_dict.items():
            # Create a list of possible key names to try
            candidates = [
                k,                                         # 1. Try exact match
                f"encoder.{k}",                            # 2. Try adding one 'encoder.' prefix
                f"encoder.encoder.{k}",                    # 3. Try adding two 'encoder.' prefixes
                k.replace("encoder.", "encoder.encoder.")  # 4. Try injecting nested encoder
            ]
            
            match_found = False
            for cand in candidates:
                if cand in expected_keys:
                    # Verify shape matches to be safe
                    if model_state_dict[cand].shape == v.shape:
                        fixed_state_dict[cand] = v
                        match_found = True
                        break
            
            # If we couldn't find a target, keep original (will show in unexpected list)
            if not match_found:
                fixed_state_dict[k] = v

        # Load the fixed dictionary
        load_result = model.audio_encoder.load_state_dict(fixed_state_dict, strict=False)
        
        print(f"Audio Encoder loaded. Missing: {len(load_result.missing_keys)}, Unexpected: {len(load_result.unexpected_keys)}")
        if len(load_result.missing_keys) > 0:
            print(f"First 5 missing keys: {load_result.missing_keys[:5]}")
        # -------------------------------------

    elif model_cfg.get("use_weighted_layer_sum", False):
        print(
            "WARNING: use_weighted_layer_sum=True but audio_encoder_{} not found. "
            "Layer-pooler weights are untrained (init). Re-train with updated train_tokens.py to save them.".format(epoch)
        )

    # Load LoRA or Full Model
    use_lora = config["model"].get("lora", True)
    if use_lora:
        lora_path = os.path.join(checkpoint_dir, f"lora_{epoch}")
        if os.path.exists(lora_path):
            print(f"Loading LoRA weights from {lora_path}")
            model.llm_model.load_adapter(lora_path, adapter_name="default")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            # Manual loading with key renaming to avoid shape mismatch (Fairseq vs HF, encoder nesting, LLM wrapping)
            model_state_dict = model.state_dict()
            pretrained_dict = {}
            shape_mismatch_keys = []
            for k, v in checkpoint.items():
                candidates = [k]
                if "audio_encoder.encoder.encoder." in k:
                    candidates.append(k.replace("audio_encoder.encoder.encoder.", "audio_encoder.encoder."))
                if "audio_encoder.encoder." in k and "audio_encoder.encoder.encoder." not in k:
                    candidates.append(k.replace("audio_encoder.encoder.", "audio_encoder.encoder.encoder."))
                if "audio_encoder.layers." in k:
                    candidates.append(k.replace("audio_encoder.layers.", "audio_encoder.encoder.layers."))
                if "audio_encoder.encoder.layers." in k:
                    candidates.append(k.replace("audio_encoder.encoder.layers.", "audio_encoder.layers."))
                if "llm_model.base_model.model.model." in k:
                    candidates.append(k.replace("llm_model.base_model.model.model.", "llm_model.model."))
                match_found = False
                for cand in candidates:
                    if cand in model_state_dict:
                        if v.shape == model_state_dict[cand].shape:
                            pretrained_dict[cand] = v
                            match_found = True
                            break
                        else:
                            shape_mismatch_keys.append(cand)
                            logger.warning(
                                f"Shape mismatch for {cand} (from {k}): checkpoint {v.shape} vs model {model_state_dict[cand].shape}"
                            )
                if not match_found and "audio_encoder" in k:
                    pass
            model.load_state_dict(pretrained_dict, strict=False)
            print(f"Checkpoint loaded (partial). Shape mismatches ignored: {len(shape_mismatch_keys)}")
            if shape_mismatch_keys:
                print(f"First 5 mismatched keys: {shape_mismatch_keys[:5]}")
            if any("speech_proj" in k for k in pretrained_dict.keys()):
                print("Confirmed: speech_proj weights loaded from checkpoint.")
            else:
                print("WARNING: speech_proj weights NOT loaded from checkpoint!")
            if any("audio_encoder" in k for k in pretrained_dict.keys()):
                print("Confirmed: audio_encoder weights loaded from checkpoint.")
            else:
                print("WARNING: audio_encoder weights NOT loaded from checkpoint!")

    # Move to device
    model.to(device)
    model.eval()
    
    # Enable CUDA optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    # Compile model (PyTorch 2.0+) - do this BEFORE mixed precision
    if hasattr(torch, 'compile') and compile_model:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="max-autotune")
    
    return model, amp_dtype if use_amp else None

@torch.inference_mode()
def evaluate_batch(model, batch, device, real_token_id, fake_token_id, amp_dtype=None):
    """Evaluate a single batch efficiently"""
    # Move to device
    raw_wav = batch["raw_wav"].to(device, non_blocking=True)
    audio_attention_mask = batch.get("audio_attention_mask")
    if audio_attention_mask is not None:
        audio_attention_mask = audio_attention_mask.to(device, non_blocking=True)
    
    prompts = batch["prompt"]
    
    samples = {
        "raw_wav": raw_wav,
        "audio_attention_mask": audio_attention_mask,
        "prompt": prompts
    }
    
    # Use autocast for mixed precision if specified
    if amp_dtype == 'fp16':
        autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16)
    elif amp_dtype == 'bf16':
        autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()
    
    with autocast_ctx:
        # Get logits (single forward pass)
        logits = model.get_last_token_logits(samples, prompts=prompts)
        
        # Extract only relevant token logits
        logits_real = logits[:, real_token_id]
        logits_fake = logits[:, fake_token_id]
        
        # Compute probabilities only for real/fake tokens
        logits_subset = torch.stack([logits_fake, logits_real], dim=1)  # [B, 2]
        probs = torch.softmax(logits_subset.float(), dim=-1)  # Convert to float32 for stability
        
        # Get predictions (argmax)
        pred_labels = probs[:, 1] > probs[:, 0]  # 1 if Real, 0 if Fake
    
    return {
        'pred_labels': pred_labels.cpu(),
        'probs_real': probs[:, 1].cpu(),
        'probs_fake': probs[:, 0].cpu(),
        'logits_real': logits_real.float().cpu(),
        'logits_fake': logits_fake.float().cpu()
    }


@torch.inference_mode()
def run_one_generate_and_log(model, batch, device, amp_dtype=None, num_samples=8):
    """Run generate once on the first num_samples and log raw output to verify no hallucination."""
    n = min(num_samples, batch["raw_wav"].shape[0])
    raw_wav = batch["raw_wav"][:n].to(device, non_blocking=True)
    audio_attention_mask = batch.get("audio_attention_mask")
    if audio_attention_mask is not None:
        audio_attention_mask = audio_attention_mask[:n].to(device, non_blocking=True)
    prompts = list(batch["prompt"][:n])
    samples = {
        "raw_wav": raw_wav,
        "audio_attention_mask": audio_attention_mask,
        "prompt": prompts
    }
    if amp_dtype == 'fp16':
        autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16)
    elif amp_dtype == 'bf16':
        autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()
    with autocast_ctx:
        out = model.generate(
            samples,
            generate_cfg={"max_new_tokens": 20, "do_sample": False},
            prompts=prompts
        )
    for i, raw_text in enumerate(out):
        text = raw_text.strip() if raw_text else ""
        logger.info(
            "[sanity generate] sample %d: raw model output = %r (check Real/Fake, not hallucination)",
            i + 1, text
        )


def main(args):
    # Determine output paths (same logic as at save time)
    if args.output_csv:
        output_csv_path = args.output_csv
    else:
        input_name = os.path.splitext(os.path.basename(args.input_jsonl))[0]
        output_csv_path = os.path.join(args.checkpoint_dir, f"NEW{input_name}_{args.epoch}_42.csv")
    metrics_path = os.path.splitext(output_csv_path)[0] + f"NEW_metrics_{args.epoch}_42.json"

    if os.path.isfile(output_csv_path) and os.path.isfile(metrics_path):
        print(f"Outputs already exist, skipping evaluation: {output_csv_path}, {metrics_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable TF32 for better performance on Ampere GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    
    # Load Model
    model, amp_dtype = load_model(
        args.checkpoint_dir, 
        args.epoch, 
        device, 
        compile_model=args.compile,
        use_amp=args.mixed_precision is not None,
        amp_dtype=args.mixed_precision
    )
    
    # Load Dataset
    dataset = AudioDeepfakeDataset(args.input_jsonl, prompt_path=None)
    if args.prompt:
        dataset.prompts = [args.prompt]
    
    # Optimized DataLoader settings
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Pre-compute token IDs
    token_real = model.tokenizer.encode("Real", add_special_tokens=False)[0]
    token_fake = model.tokenizer.encode("Fake", add_special_tokens=False)[0]
    print(f"Token IDs: Real={token_real}, Fake={token_fake}")
    print(f"Using AMP: {amp_dtype if amp_dtype else 'No (FP32)'}")
    
    # Pre-allocate lists
    total_samples = len(dataset)
    y_true = []
    y_pred = []
    y_scores_real = []
    results = []
    
    print(f"Starting evaluation on {total_samples} samples...")
    print(f"Batch size: {args.batch_size}, Workers: {args.num_workers}")
    sanity_generate_done = False

    # Main evaluation loop
    for batch in tqdm(dataloader, disable=args.disable_tqdm):
        if args.log_one_generate and not sanity_generate_done:
            run_one_generate_and_log(model, batch, device, amp_dtype)
            sanity_generate_done = True
        batch_results = evaluate_batch(
            model, batch, device, token_real, token_fake, amp_dtype
        )
        
        pred_labels = batch_results['pred_labels']
        probs_real = batch_results['probs_real']
        probs_fake = batch_results['probs_fake']
        logits_real = batch_results['logits_real']
        logits_fake = batch_results['logits_fake']
        
        # Process results
        for i in range(len(batch["path"])):
            gt_text = batch["text"][i]
            
            # Determine GT label
            gt_lower = gt_text.lower()
            if "real" in gt_lower:
                gt_label = 1
            elif "fake" in gt_lower:
                gt_label = 0
            else:
                gt_label = -1
            
            pred_label = int(pred_labels[i])
            
            if gt_label != -1:
                y_true.append(gt_label)
                y_pred.append(pred_label)
                y_scores_real.append(float(probs_real[i]))
            
            results.append({
                "path": batch["path"][i],
                "ground_truth": gt_text,
                "prediction": "Real" if pred_label == 1 else "Fake",
                "prompt": batch["prompt"][i],
                "score_real": float(probs_real[i]),
                "score_fake": float(probs_fake[i]),
                "logits_real": float(logits_real[i]),
                "logits_fake": float(logits_fake[i]),
            })
    
    # Convert to numpy for faster computation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores_real = np.array(y_scores_real)
    
    # Calculate metrics
    accuracy = np.mean(y_true == y_pred)
    
    # Fake metrics (label 0)
    fake_mask_true = (y_true == 0)
    fake_mask_pred = (y_pred == 0)
    
    tp_fake = np.sum(fake_mask_true & fake_mask_pred)
    tn_fake = np.sum(~fake_mask_true & ~fake_mask_pred)
    fp_fake = np.sum(~fake_mask_true & fake_mask_pred)
    fn_fake = np.sum(fake_mask_true & ~fake_mask_pred)
    
    precision = tp_fake / (tp_fake + fp_fake) if (tp_fake + fp_fake) > 0 else 0
    recall = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    fpr, fnr, thresholds = det_curve(y_true, y_scores_real)
    index = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[index] + fnr[index]
    eer = eer/2

    metrics = {
        "total_samples": len(results),
        "accuracy": float(accuracy),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "eer": float(eer),
        "tp_fake": int(tp_fake),
        "tn_fake": int(tn_fake),
        "fp_fake": int(fp_fake),
        "fn_fake": int(fn_fake),
        "num_fake_gt": int(np.sum(y_true == 0)),
        "num_real_gt": int(np.sum(y_true == 1))
    }
    
    print("="*30)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("="*30)
    
    # Save results (output_csv_path and metrics_path set at start of main)
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--epoch", type=str, default="best")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8, help="Increase for faster inference")
    parser.add_argument("--num_workers", type=int, default=8, help="Increase for faster data loading")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=[None, 'fp16', 'bf16'], help="Use AMP (recommended: bf16)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for speedup")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--no_log_one_generate", action="store_true", help="Disable one-call generate sanity check (no raw output logged).")
    parser.add_argument("--prompt", type=str, default="Listen to this audio clip: <Speech><SpeechHere></Speech>. Is this audio real or fake?")
    args = parser.parse_args()
    args.log_one_generate = not args.no_log_one_generate
    main(args)