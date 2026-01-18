import os
import argparse
import yaml
import torch
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, det_curve
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.audio_llm import AudioDeepfakeLLM
from dataset import AudioDeepfakeDataset

from huggingface_hub import login
login(token="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")

def load_model(checkpoint_dir, epoch, device):
    # Load Config
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found in {checkpoint_dir}. Expected config.json")
    
    # Initialize Model
    model_cfg = config["model"]
    model = AudioDeepfakeLLM(
        llm_path=model_cfg["llm_path"],
        llm_type=model_cfg.get("llm_type", "qwen2"),
        audio_encoder_path=model_cfg["audio_encoder_path"],
        audio_encoder_type=model_cfg.get("audio_encoder_type", "whisper"),
        audio_encoder_source=model_cfg.get("audio_encoder_source", "huggingface"),
        freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True),
        lora=model_cfg.get("lora", True),
        lora_rank=model_cfg.get("lora_rank", 64),
        lora_alpha=model_cfg.get("lora_alpha", 16),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        special_tokens=model_cfg.get("special_tokens", []),
        device_8bit=0 
    )
    
    # Load Projector Weights
    if epoch == "best":
        proj_path = os.path.join(checkpoint_dir, f"proj_best.pt")
    else:
        proj_path = os.path.join(checkpoint_dir, f"proj_epoch_{epoch}.pt")
    if os.path.exists(proj_path):
        print(f"Loading projector weights from {proj_path}")
        model.speech_proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
    else:
        print(f"Warning: Projector weights not found at {proj_path}")

    # Load Audio Encoder Weights if they exist
    if epoch == "best":
        audio_enc_path = os.path.join(checkpoint_dir, f"audio_encoder_best.pt")
    else:
        audio_enc_path = os.path.join(checkpoint_dir, f"audio_encoder_epoch_{epoch}.pt")
    if os.path.exists(audio_enc_path):
        print(f"Loading audio encoder weights from {audio_enc_path}")
        model.audio_encoder.load_state_dict(torch.load(audio_enc_path, map_location="cpu"))

    # Load LoRA Weights
    if epoch == "best":
        lora_path = os.path.join(checkpoint_dir, f"lora_best")
    else:
        lora_path = os.path.join(checkpoint_dir, f"lora_epoch_{epoch}")
    if os.path.exists(lora_path) and config["model"].get("lora", True):
        print(f"Loading LoRA weights from {lora_path}")
        model.llm_model.load_adapter(lora_path, adapter_name="default")
    else:
        print(f"Warning: LoRA weights not found at {lora_path} or LoRA disabled.")

    model.to(device)
    model.eval()
    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = load_model(args.checkpoint_dir, args.epoch, device)
    
    # Load Dataset
    # We use AudioDeepfakeDataset but we might want to override prompts to be fixed for evaluation
    dataset = AudioDeepfakeDataset(args.input_jsonl, prompt_path=None)
    
    # Override prompts to use a single fixed prompt for consistency if provided, 
    # otherwise dataset uses random from list.
    if args.prompt:
        dataset.prompts = [args.prompt]
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=dataset.collate_fn,
        num_workers=4
    )
    
    results = []
    
    # Metrics tracking
    y_true = []
    y_pred = []
    y_scores_real = [] # Probabilities of Real class
    hallucination_count = 0
    total_samples = 0
    
    # Identify token IDs for "Real" and "Fake"
    token_real = model.tokenizer.encode("Real", add_special_tokens=False)
    token_fake = model.tokenizer.encode("Fake", add_special_tokens=False)
    
    # Use the first token if multiple (though for "Real" it should be one)
    real_token_id = token_real[0]
    fake_token_id = token_fake[0]
    print(f"Using Token IDs: Real={real_token_id} ({model.tokenizer.decode([real_token_id])}), Fake={fake_token_id} ({model.tokenizer.decode([fake_token_id])})")
    
    print(f"Starting evaluation on {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            raw_wav = batch["raw_wav"].to(device)
            audio_attention_mask = batch.get("audio_attention_mask", None)
            if audio_attention_mask is not None:
                audio_attention_mask = audio_attention_mask.to(device)
            
            # Use the prompts from the batch (which are strings)
            prompts = batch["prompt"]
            
            samples = {
                "raw_wav": raw_wav,
                "audio_attention_mask": audio_attention_mask,
                "prompt": prompts
            }
            
            # 1. Generate Text (for Accuracy/F1)
            outputs = model.generate(samples, generate_cfg={"max_new_tokens": 50, "do_sample": False})
            
            # 2. Get Probabilities (for EER)
            logits = model.get_last_token_logits(samples, prompts=prompts)
            probs = torch.softmax(logits, dim=-1) # [B, V]
            
            batch_real_probs = probs[:, real_token_id].cpu().tolist()
            batch_fake_probs = probs[:, fake_token_id].cpu().tolist()
            
            # Collect results
            for i in range(len(outputs)):
                path = batch["path"][i]
                gt_text = batch["text"][i]
                pred_text = outputs[i]
                prompt = prompts[i]
                
                # EER Score Calculation
                p_real = batch_real_probs[i]
                p_fake = batch_fake_probs[i]
                                
                # Metric Calculation
                total_samples += 1
                
                # Determine GT Label (1=Real, 0=Fake)
                if "real" in gt_text.lower():
                    gt_label = 1
                elif "fake" in gt_text.lower():
                    gt_label = 0
                else:
                    gt_label = -1 
                
                # Determine Pred Label
                pred_lower = pred_text.lower()
                is_pred_real = "real" in pred_lower and "fake" not in pred_lower
                is_pred_fake = "fake" in pred_lower and "real" not in pred_lower
                
                if is_pred_real:
                    pred_label = 1
                elif is_pred_fake:
                    pred_label = 0
                else:
                    pred_label = -1 # Hallucination
                    hallucination_count += 1
                
                if gt_label != -1:
                    y_true.append(gt_label)
                    y_pred.append(pred_label)
                    y_scores_real.append(p_real)

                results.append({
                    "path": path,
                    "ground_truth": gt_text,
                    "prediction": pred_text,
                    "prompt": prompt,
                    "hallucination": pred_label == -1,
                    "score_real": p_real,
                    "score_fake": p_fake,
                })
                print(f"Pred: {pred_text} | GT: {gt_text} | Score(Real): {p_real:.6f}")

    accuracy = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0
    # Calculate Precision/Recall/F1 for Fake (Label 0)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0], average='micro', zero_division=0)
    

    # EER Calculation
    eer = 0.0
    if len(y_true) > 0:
        fpr, fnr, thresholds = det_curve(y_true, y_scores_real)
        index = np.nanargmin(np.absolute(fnr - fpr))
        eer = fpr[index] + fnr[index]
        eer = eer/2

    hallucination_rate = hallucination_count / total_samples if total_samples > 0 else 0
    
    cm_labels = [0, 1] 

    # Update counts for Real=1, Fake=0
    # TP (Fake) = GT=0, Pred=0
    tp_fake = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    tn_fake = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp_fake = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    fn_fake = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp != 0) 
    
    num_fake_gt = sum(1 for x in y_true if x == 0)
    num_real_gt = sum(1 for x in y_true if x == 1)

    metrics = {
        "total_samples": total_samples,
        "hallucinations": hallucination_count,
        "hallucination_rate": hallucination_rate,
        "accuracy": accuracy,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "eer": eer,
        "tp_fake": tp_fake,
        "tn_fake": tn_fake,
        "fp_fake": fp_fake,
        "fn_fake": fn_fake,
        "num_fake_gt": num_fake_gt,
        "num_real_gt": num_real_gt
    }
    
    print("="*30)
    print("Evaluation Metrics:")
    print(f"Total Samples: {total_samples}")
    print(f"Hallucinations: {hallucination_count} ({hallucination_rate:.2%})")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Fake): {precision:.4f}")
    print(f"Recall (Fake): {recall:.4f}")
    print(f"F1 Score (Fake): {f1:.4f}")
    print(f"EER: {eer:.4f}")
    print("="*30)

    # Determine output CSV path based on input JSONL name
    if args.output_csv:
        output_csv_path = args.output_csv
    else:
        # Extract base name from input JSONL (e.g., "itw.jsonl" -> "itw")
        input_basename = os.path.basename(args.input_jsonl)
        input_name = os.path.splitext(input_basename)[0]
        # Save in checkpoint directory
        output_csv_path = os.path.join(args.checkpoint_dir, f"{input_name}_{args.epoch}.csv")
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
    
    # Save Metrics JSON (in same directory as CSV)
    base_name = os.path.splitext(output_csv_path)[0]
    metrics_path = base_name + f"_metrics_{args.epoch}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True, help="Path to config yaml") # Removed
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoints (output_dir)")
    parser.add_argument("--epoch", type=str, required=False, default="best", help="Epoch number to load")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to output CSV file (optional, defaults to <input_name>.csv in checkpoint_dir)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--prompt", type=str, default="Listen to this audio clip: <Speech><SpeechHere></Speech>. Is this audio real or fake?", help="Prompt to use for evaluation")
    
    args = parser.parse_args()
    main(args)
