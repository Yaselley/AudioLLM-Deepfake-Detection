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
from sklearn.metrics import accuracy_score, det_curve
from peft import PeftModel  # REQUIRED for loading your trained adapters

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_clean.qwen_opensmile import AudioDeepfakeLLM
from dataset_qwen import AudioDeepfakeDataset
from huggingface_hub import login

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
HF_TOKEN = "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"
BASE_MODEL_ID = "Qwen/Qwen2.5-Omni-3B" # The base model you trained on

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")

def load_model_for_eval(base_model_path, checkpoint_dir, device, use_amp=False, amp_dtype='fp16'):
    """
    Loads the Base Model and applies the LoRA adapter from the checkpoint.
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    logger.info(f"Step 1: Loading BASE model: {base_model_path}")
    
    # 1. Initialize the wrapper with the BASE model
    # We set use_lora=False here because we don't want to initialize NEW random LoRA weights.
    # We want to load your PRE-TRAINED ones.
    model = AudioDeepfakeLLM(
        model_path=base_model_path,
        use_lora=False, 
        low_resource=False
    )
    
    logger.info(f"Step 2: Loading LoRA adapters from: {checkpoint_dir}")
    
    # 2. Load the trained LoRA adapter on top of the base model
    # model.model is the actual Qwen2_5OmniThinkerForConditionalGeneration
    try:
        model.model = PeftModel.from_pretrained(model.model, checkpoint_dir)
        
        # 3. Merge weights for faster inference (optional but recommended for eval)
        logger.info("Merging LoRA weights into base model for efficiency...")
        model.model = model.model.merge_and_unload()
    except Exception as e:
        logger.warning(f"Could not load/merge LoRA adapters: {e}")
        logger.warning("Proceeding with Base Model (or full model if checkpoint was full save).")

    model.to(device)
    model.eval()
    
    return model, amp_dtype if use_amp else None

@torch.inference_mode()
def evaluate_batch(model, batch, device, real_token_id, fake_token_id, amp_dtype=None):
    """Evaluate a single batch efficiently"""
    samples = {
        "raw_wav": batch["raw_wav"],
        "text": batch["text"],
        "prompt": batch["prompt"]
    }
    
    # Determine Autocast Context
    if amp_dtype == 'fp16':
        dtype = torch.float16
    elif amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = None
        
    autocast_ctx = torch.amp.autocast('cuda', dtype=dtype) if dtype else nullcontext()
    
    with autocast_ctx:
        # Get logits for the LAST token (next token prediction)
        logits = model.get_last_token_logits(samples)
        
        # Extract logits for specific tokens [Batch, 2] -> [Fake, Real]
        logits_real = logits[:, real_token_id]
        logits_fake = logits[:, fake_token_id]
        
        # Stack: [Fake, Real]
        logits_subset = torch.stack([logits_fake, logits_real], dim=1)
        
        # Softmax to get probabilities
        probs = torch.softmax(logits_subset.float(), dim=-1)
        
        # Predict: 1 if Real > Fake, else 0
        pred_labels = probs[:, 1] > probs[:, 0] 
    
    return {
        'pred_labels': pred_labels.cpu(),     # 1=Real, 0=Fake
        'probs_real': probs[:, 1].cpu(),      # Probability of Real
        'probs_fake': probs[:, 0].cpu(),      # Probability of Fake
        'logits_real': logits_real.float().cpu(),
        'logits_fake': logits_fake.float().cpu()
    }

def get_clean_token_id(tokenizer, word):
    """
    Get the token ID for a word without leading spaces.
    Crucial for ChatML format where prompts end in newline.
    """
    # 1. Try direct vocabulary lookup
    vocab = tokenizer.get_vocab()
    if word in vocab:
        return vocab[word]
    
    # 2. Fallback: Encode and strip
    # Some tokenizers add a start token or space automatically
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
        
    logger.warning(f"Word '{word}' split into multiple tokens: {ids}. Using the first one.")
    return ids[0]

def main(args):
    set_seed(42)
    login(token=HF_TOKEN)

    # Output paths
    if args.output_csv:
        output_csv_path = args.output_csv
    else:
        input_name = os.path.splitext(os.path.basename(args.input_jsonl))[0]
        checkpoint_name = os.path.basename(args.checkpoint_dir.rstrip('/'))
        output_csv_path = os.path.join(
            args.checkpoint_dir, 
            f"{input_name}_{checkpoint_name}_results.csv"
        )
    
    metrics_path = os.path.splitext(output_csv_path)[0] + "_metrics.json"

    if os.path.isfile(output_csv_path) and os.path.isfile(metrics_path) and not args.overwrite:
        logger.info(f"Outputs exist. Skipping. Use --overwrite to regenerate.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === 1. LOAD MODEL CORRECTLY ===
    model, amp_dtype = load_model_for_eval(
        args.base_model,      # Pass the base model (Qwen repo)
        args.checkpoint_dir,  # Pass your trained adapter folder
        device,
        use_amp=args.mixed_precision is not None,
        amp_dtype=args.mixed_precision
    )
    
    # === 2. FIX TOKENIZATION ===
    # Qwen ChatML ends prompt with "\n". The next token is the start of the word.
    # We must ensure we get the ID for "Real", NOT " Real" (with space).
    tokenizer = model.processor.tokenizer
    token_real = get_clean_token_id(tokenizer, "Real")
    token_fake = get_clean_token_id(tokenizer, "Fake")
    
    logger.info(f"Target Token IDs: Real={token_real}, Fake={token_fake}")
    logger.info(f"decoded check: {tokenizer.decode([token_real])} vs {tokenizer.decode([token_fake])}")

    # Load Dataset
    logger.info(f"Loading dataset from {args.input_jsonl}...")
    dataset = AudioDeepfakeDataset(
        args.input_jsonl, 
        prompt_path=args.prompt_path
    )
    
    if args.prompt:
        dataset.prompts = [args.prompt]
        logger.info(f"Using custom prompt: {args.prompt}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Storage
    y_true = []
    y_pred = []
    y_scores_real = [] # Probability of being Real
    results = []
    
    logger.info(f"Starting evaluation on {len(dataset)} samples...")
    
    # === Main Evaluation Loop ===
    # model.eval()
    # for batch_idx, batch in enumerate(tqdm(dataloader, disable=args.disable_tqdm)):
    #     try:
    #         batch_results = evaluate_batch(
    #             model, batch, device, token_real, token_fake, amp_dtype
    #         )
    #     except Exception as e:
    #         logger.error(f"Error processing batch {batch_idx}: {e}")
    #         continue
        
    #     pred_labels = batch_results['pred_labels'].numpy()
    #     probs_real = batch_results['probs_real'].numpy()
    #     probs_fake = batch_results['probs_fake'].numpy()
    #     logits_real = batch_results['logits_real'].numpy()
    #     logits_fake = batch_results['logits_fake'].numpy()
        
    #     # Match predictions to Ground Truth
    #     current_batch_size = len(pred_labels)
    #     start_idx = batch_idx * args.batch_size
        
    #     for i in range(current_batch_size):
    #         # Safe dataset access
    #         dataset_idx = start_idx + i
    #         if dataset_idx >= len(dataset):
    #             break
            
    #         # 1. Parse Ground Truth
    #         gt_text = batch["text"][i]
    #         gt_lower = gt_text.lower()
    #         if "real" in gt_lower:
    #             gt_label = 1
    #         elif "fake" in gt_lower:
    #             gt_label = 0
    #         else:
    #             gt_label = -1 # Invalid
            
    #         # 2. Store for Metrics
    #         if gt_label != -1:
    #             y_true.append(gt_label)
    #             y_pred.append(pred_labels[i])
    #             y_scores_real.append(probs_real[i])
            
    #         # 3. Get Metadata
    #         try:
    #             data_item = dataset.data[dataset_idx]
    #             file_path = data_item.get("path", "unknown")
    #         except (AttributeError, IndexError):
    #             file_path = f"sample_{dataset_idx}"

    #         # 4. Store Result Row
    #         results.append({
    #             "path": file_path,
    #             "ground_truth_text": gt_text,
    #             "ground_truth_label": gt_label,
    #             "prediction": "Real" if pred_labels[i] == 1 else "Fake",
    #             "score_real": float(probs_real[i]),
    #             "score_fake": float(probs_fake[i]),
    #             "logit_real": float(logits_real[i]),
    #             "logit_fake": float(logits_fake[i]),
    #             "prompt": batch["prompt"][i]
    #         })
    # === Main Evaluation Loop ===
    model.eval()
    logger.info("Starting evaluation...")

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=args.disable_tqdm)):
        
        # ====================================================
        # NEW: Generate and Print First Batch for Inspection
        # ====================================================
        if batch_idx == 0:
            logger.info("Generating text for the first batch to inspect model behavior...")
            
            # Prepare samples for the wrapper's generate method
            gen_samples = {
                "raw_wav": batch["raw_wav"],
                "text": batch["text"],    # Ground truth (used for comparison)
                "prompt": batch["prompt"] # Input prompt
            }

            # Generate (Force deterministic generation for inspection)
            # max_new_tokens=20 is usually enough for "Real/Fake" + explanation
            try:
                generated_texts = model.generate(
                    gen_samples, 
                    max_new_tokens=30, 
                    do_sample=False
                )

                print("\n" + "!"*20 + " FIRST BATCH INSPECTION " + "!"*20)
                for i, (gen_text, true_text) in enumerate(zip(generated_texts, batch['text'])):
                    print(f"\n[Sample {i}]")
                    print(f"  Prompt end:   ...{batch['prompt'][i][-50:]!r}") # Show end of prompt
                    print(f"  Ground Truth: {true_text}")
                    print(f"  Model Output: \033[92m{gen_text.strip()}\033[0m") # Green text for visibility
                print("\n" + "!"*65 + "\n")
            except Exception as e:
                logger.error(f"Generation failed: {e}")

        # ====================================================
        # EXISTING: Logit Evaluation
        # ====================================================
        try:
            batch_results = evaluate_batch(
                model, batch, device, token_real, token_fake, amp_dtype
            )
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            continue
        
        # ... (Rest of your metrics collection code remains the same) ...
        pred_labels = batch_results['pred_labels'].numpy()
        probs_real = batch_results['probs_real'].numpy()
        probs_fake = batch_results['probs_fake'].numpy()
        logits_real = batch_results['logits_real'].numpy()
        logits_fake = batch_results['logits_fake'].numpy()
        
        # Match predictions to Ground Truth
        current_batch_size = len(pred_labels)
        start_idx = batch_idx * args.batch_size
        
        for i in range(current_batch_size):
            # Safe dataset access
            dataset_idx = start_idx + i
            if dataset_idx >= len(dataset):
                break
            
            # 1. Parse Ground Truth
            gt_text = batch["text"][i]
            gt_lower = gt_text.lower()
            if "real" in gt_lower:
                gt_label = 1
            elif "fake" in gt_lower:
                gt_label = 0
            else:
                gt_label = -1 # Invalid
            
            # 2. Store for Metrics
            if gt_label != -1:
                y_true.append(gt_label)
                y_pred.append(pred_labels[i])
                y_scores_real.append(probs_real[i])
            
            # 3. Get Metadata
            try:
                data_item = dataset.data[dataset_idx]
                file_path = data_item.get("path", "unknown")
            except (AttributeError, IndexError):
                file_path = f"sample_{dataset_idx}"

            # 4. Store Result Row
            results.append({
                "path": file_path,
                "ground_truth_text": gt_text,
                "ground_truth_label": gt_label,
                "prediction": "Real" if pred_labels[i] == 1 else "Fake",
                "score_real": float(probs_real[i]),
                "score_fake": float(probs_fake[i]),
                "logit_real": float(logits_real[i]),
                "logit_fake": float(logits_fake[i]),
                "prompt": batch["prompt"][i],
                # Add generated text to CSV if it was the first batch, else empty
                "generated_text": generated_texts[i] if batch_idx == 0 else "" 
            })
    # === Metrics Calculation ===
    if not y_true:
        logger.error("No valid samples found for evaluation!")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores_real = np.array(y_scores_real)
    
    # 1. Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # 2. Fake Detection Metrics (Target Class = 0)
    fake_mask_true = (y_true == 0)
    fake_mask_pred = (y_pred == 0)
    
    tp_fake = np.sum(fake_mask_true & fake_mask_pred)
    tn_fake = np.sum(~fake_mask_true & ~fake_mask_pred)
    fp_fake = np.sum(~fake_mask_true & fake_mask_pred)
    fn_fake = np.sum(fake_mask_true & ~fake_mask_pred)
    
    precision = tp_fake / (tp_fake + fp_fake) if (tp_fake + fp_fake) > 0 else 0.0
    recall = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 3. EER (Equal Error Rate)
    fpr, fnr, thresholds = det_curve(y_true, y_scores_real)
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2

    metrics = {
        "accuracy": float(accuracy),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "eer": float(eer),
        "tp_fake": int(tp_fake),
        "tn_fake": int(tn_fake),
        "fp_fake": int(fp_fake),
        "fn_fake": int(fn_fake),
        "total_samples": int(len(y_true))
    }
    
    # === Reporting ===
    print("\n" + "="*50)
    print("Evaluation Metrics:")
    print("="*50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")
    print("="*50 + "\n")
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Results saved to {output_csv_path}")
    
    # Save Metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen-based Audio Deepfake Detection Model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                        help="Path to saved LoRA adapter directory (e.g., ./output/best_model)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Omni-3B",
                        help="The base model to load adapters onto")
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Path to evaluation JSONL file")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to output CSV file")
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", 
                        choices=[None, 'fp16', 'bf16'])
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    
    args = parser.parse_args()
    main(args)