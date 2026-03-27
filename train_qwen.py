import os
import argparse
import yaml
import logging
import torch
import json
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from huggingface_hub import login
import numpy as np
import random

from models_clean.qwen_opensmile import AudioDeepfakeLLM
from dataset_qwen import AudioDeepfakeDataset

import warnings
# Ignore the specific dependency warning
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger().setLevel(logging.ERROR)

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir):
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            logging.StreamHandler()
        ]
    )


def count_parameters(model):
    """Count trainable params safely handling PEFT structure."""
    audio_params = 0
    llm_params = 0
    total_trainable = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            total_trainable += param.numel()
            if 'audio' in name.lower():
                audio_params += param.numel()
            else:
                llm_params += param.numel()
    
    return {
        'audio_params': audio_params,
        'llm_params': llm_params,
        'total_trainable': total_trainable,
        'total_params': total_params
    }


def main(args):
    # Authenticate
    if args.hf_token:
        login(token=args.hf_token)
    else:
        login(token="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")

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = config.get("output_dir", "./output_qwenomni")
    setup_logging(output_dir)
    logging.info(f"Config: {config}")
    
    # Set Seed
    set_seed(config["train"].get("seed", 42))
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Model
    model_cfg = config["model"]
    model = AudioDeepfakeLLM(
        model_path=model_cfg.get("model_path", "Qwen/Qwen2.5-Omni-3B-Instruct"),
        lora_rank=model_cfg.get("lora_rank", 64),
        lora_alpha=model_cfg.get("lora_alpha", 16),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
    )
    # Note: Model is already moved to device inside its __init__ (device_map="auto")
    # or we can force it here just in case parts are on CPU
    if not config["model"].get("low_resource", False):
        model.to(device)
    
    # Count parameters
    param_counts = count_parameters(model)
    logging.info("="*50)
    logging.info(f"Parameter counts:")
    logging.info(f"  Audio Encoder (trainable): {param_counts['audio_params']:,}")
    logging.info(f"  LLM (trainable):           {param_counts['llm_params']:,}")
    logging.info(f"  Total trainable:           {param_counts['total_trainable']:,}")
    logging.info(f"  Total parameters:          {param_counts['total_params']:,}")
    logging.info("="*50)
    
    # Dataset
    logging.info("Loading Datasets...")
    train_dataset = AudioDeepfakeDataset(config["data"]["train_path"], prompt_path=config["data"].get("prompt_path"))
    val_dataset = AudioDeepfakeDataset(config["data"]["val_path"], prompt_path=config["data"].get("prompt_path"))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=True, 
        collate_fn=train_dataset.collate_fn,
        num_workers=config["train"].get("num_workers", 4)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=False, 
        collate_fn=val_dataset.collate_fn,
        num_workers=config["train"].get("num_workers", 4)
    )
    
    # Optimizer
    # We only optimize parameters that have requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params, 
        lr=float(config["train"].get("lr", 1e-4)),
        weight_decay=config["train"].get("weight_decay", 0.05)
    )
    
    num_epochs = config["train"]["epochs"]
    grad_accum_steps = config["train"].get("grad_accum_steps", 1)
    
    # Training Loop
    logging.info("Starting Training...")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            # CRITICAL FIX: Do NOT move raw_wav to device here. 
            # It is a list of numpy arrays. The model handles device movement.
            
            samples = {
                "raw_wav": batch["raw_wav"], # Pass as list
                "text": batch["text"],
                "prompt": batch["prompt"]
            }
            
            # Forward pass
            outputs = model(samples)
            loss = outputs["loss"] / grad_accum_steps
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            total_loss += loss.item() * grad_accum_steps
            
            if step % 1000 == 0:
                logging.info(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item() * grad_accum_steps:.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} Complete. Avg Train Loss: {avg_train_loss:.4f} | Time: {time.time() - start_time:.2f}s")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                samples = {
                    "raw_wav": batch["raw_wav"],
                    "text": batch["text"],
                    "prompt": batch["prompt"]
                }
                outputs = model(samples)
                val_loss += outputs["loss"].item()
        
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logging.info(f"✓ New Best Model! Saving...")
            
            save_path = os.path.join(output_dir, "best_model")
            model.model.save_pretrained(save_path)
            model.processor.save_pretrained(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()
    main(args)