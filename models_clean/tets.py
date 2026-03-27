import os
import argparse
import yaml
import logging
import torch
import json
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import time
from huggingface_hub import login
import numpy as np
import random
from sklearn.metrics import accuracy_score

from models_clean.audiollm_base_opensmile import AudioDeepfakeLLM
from dataset import AudioDeepfakeDataset

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(output_dir):
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

def configure_optimizer(model, config, stage):
    """
    Configure optimizer based on the training stage.
    Stage 1: Train Encoder + Adapter, Freeze LLM/LoRA
    Stage 2: Train Encoder + Adapter + LLM/LoRA
    """
    
    # 1. Set requires_grad based on stage
    # Always ensure encoder/adapter are trainable (if config allows)
    # If stage 1, freeze LLM. If stage 2, unfreeze LLM.
    
    # Check config freezing preferences for encoder
    freeze_audio_encoder = config["model"].get("freeze_audio_encoder", False) 
    
    for name, param in model.named_parameters():
        # Default: Everything trainable if not explicitly frozen by logic below
        
        if "audio_encoder" in name:
            if "layer_pooler" in name:
                 param.requires_grad = True # Always train pooler if present
            else:
                 param.requires_grad = not freeze_audio_encoder
                 
        elif "speech_proj" in name and model.speech_proj is not None:
            param.requires_grad = True # Adapter always trained
            
        else:
            # LLM / LoRA params
            if stage == 1:
                param.requires_grad = False
            else:

                if config["model"].get("lora", True):

                     if "lora_" in name or "modules_to_save" in name or "embed_tokens" in name or "lm_head" in name:
                         param.requires_grad = True
                     else:
                         # Base LLM weights -> Freeze
                         param.requires_grad = False
                else:
                    # Full finetuning?
                    param.requires_grad = True

    # 2. Collect parameters
    audio_enc_params = []
    adapter_params = []
    llm_params = []
    
    audio_enc_param_count = 0
    adapter_param_count = 0
    llm_param_count = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "audio_encoder" in name:
            audio_enc_params.append(param)
            audio_enc_param_count += param.numel()
        elif "speech_proj" in name and model.speech_proj is not None:
            adapter_params.append(param)
            adapter_param_count += param.numel()
        else:
            llm_params.append(param)
            llm_param_count += param.numel()
            
    logging.info(f"Stage {stage} Optimization Setup:")
    logging.info(f"  Audio Encoder Params:   {audio_enc_param_count:,}")
    logging.info(f"  Audio Adapter Params:   {adapter_param_count:,}")
    logging.info(f"  LLM/LoRA Params:        {llm_param_count:,}")
    logging.info(f"  Total Trainable:        {audio_enc_param_count + adapter_param_count + llm_param_count:,}")

    # 3. Create Optimizer
    # We use the same LRs as config
    params_group = []
    if audio_enc_params:
        params_group.append({'params': audio_enc_params, 'lr': float(config["train"].get("lr_audio", config["train"].get("lr", 1e-4)))})
    if adapter_params:
        params_group.append({'params': adapter_params, 'lr': float(config["train"].get("lr_adapter", config["train"].get("lr", 1e-4)))})
    if llm_params:
        params_group.append({'params': llm_params, 'lr': float(config["train"].get("lr_lora", config["train"].get("lr", 1e-4)))})
        
    optimizer = optim.AdamW(params_group, weight_decay=0.05)
    return optimizer

def main(args):
    # Authenticate with HF
    if args.hf_token:
        login(token=args.hf_token)
    else:
        login(token="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")

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = config.get("output_dir", "./output")
    setup_logging(output_dir)
    logging.info(f"Config: {config}")

    # Save Config
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Set Seed
    seed = config["train"].get("seed", 42)
    set_seed(seed)
    logging.info(f"Set seed to {seed}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Model
    logging.info("Initializing Model...")
    model_cfg = config["model"]
    
    model = AudioDeepfakeLLM(
        llm_path=model_cfg["llm_path"],
        llm_type=model_cfg.get("llm_type", "qwen2"),
        audio_encoder_path=model_cfg.get("audio_encoder_path", None),
        audio_encoder_type=model_cfg.get("audio_encoder_type", "whisper"),
        audio_encoder_source=model_cfg.get("audio_encoder_source", "huggingface"),
        freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True), # Should be False for "encoder tuned"
        # Weighted layer pooling (Whisper / Wav2Vec2)
        use_weighted_layer_sum=model_cfg.get("use_weighted_layer_sum", False),
        weighted_layer_norm=model_cfg.get("weighted_layer_norm", "none"),
        lora=model_cfg.get("lora", True),
        lora_rank=model_cfg.get("lora_rank", 64),
        lora_alpha=model_cfg.get("lora_alpha", 16),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        special_tokens=model_cfg.get("special_tokens", []),
        device_8bit=0,
        # SpeechTokenizer parameters
        speechtokenizer_num_layers=model_cfg.get("speechtokenizer_num_layers", 8),
        speechtokenizer_codebook_size=model_cfg.get("speechtokenizer_codebook_size", 1024),
        speechtokenizer_return_discrete=model_cfg.get("speechtokenizer_return_discrete", True),
        dac_num_codebooks=model_cfg.get("dac_num_codebooks", 12),
        dac_codebook_size=model_cfg.get("dac_codebook_size", 1024),
        dac_model_type=model_cfg.get("dac_model_type", "16khz"),
        dac_return_discrete=model_cfg.get("dac_return_discrete", True),
        # SQ-Codec parameters
        sqcodec_num_quantizers=model_cfg.get("sqcodec_num_quantizers", 4),
        sqcodec_num_levels=model_cfg.get("sqcodec_num_levels", 4096),

        use_opensmile=model_cfg.get("use_opensmile", False),
        opensmile_only=model_cfg.get("opensmile_only", False),
    )
    model.to(device)
    
    # Dataset
    logging.info("Loading Datasets...")
    
    train_dataset = AudioDeepfakeDataset(
        config["data"]["train_path"],
        prompt_path=config["data"].get("prompt_path", None)
    )
    val_dataset = AudioDeepfakeDataset(
        config["data"]["val_path"],
        prompt_path=config["data"].get("prompt_path", None)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=True, 
        collate_fn=train_dataset.collate_fn,
        num_workers=config["train"].get("num_workers", 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=False, 
        collate_fn=val_dataset.collate_fn,
        num_workers=config["train"].get("num_workers", 4),
        pin_memory=True
    )
    
    num_epochs = config["train"]["epochs"]
    grad_accum_steps = config["train"].get("grad_accum_steps", 1)
    
    # Training Loop
    logging.info("Starting Training with 2-Step Strategy...")
    global_step = 0
    
    # Track best validation loss
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Initialize Optimizer variable
    optimizer = None
    
    for epoch in range(num_epochs):
        # STAGE CONTROL
        if epoch == 0:
            logging.info(">>> Entering STAGE 1: Training Encoder + Adapter (Epochs 1-2)")
            optimizer = configure_optimizer(model, config, stage=1)
        elif epoch == 2:
            logging.info(">>> Entering STAGE 2: Training Encoder + Adapter + LoRA (Epochs 3+)")
            optimizer = configure_optimizer(model, config, stage=2)
            # Reset best loss tracking for the new stage as requested ("best model should be selected from here")
            best_val_loss = float('inf') 
            best_epoch = 0
            
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            raw_wav = batch["raw_wav"].to(device)
            audio_attention_mask = batch.get("audio_attention_mask", None)
            if audio_attention_mask is not None:
                audio_attention_mask = audio_attention_mask.to(device)
                
            samples = {
                "raw_wav": raw_wav,
                "audio_attention_mask": audio_attention_mask,
                "text": batch["text"],
                "prompt": batch["prompt"]
            }
            
            outputs = model(samples)
            loss = outputs["loss"] / grad_accum_steps
            
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            total_loss += loss.item() * grad_accum_steps
            
            if step % 1000 == 0:
                logging.info(f"Epoch {epoch+1}/{num_epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item() * grad_accum_steps:.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1} Complete. Avg Train Loss: {avg_train_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                raw_wav = batch["raw_wav"].to(device)
                audio_attention_mask = batch.get("audio_attention_mask", None)
                if audio_attention_mask is not None:
                    audio_attention_mask = audio_attention_mask.to(device)

                samples = {
                    "raw_wav": raw_wav,
                    "audio_attention_mask": audio_attention_mask,
                    "text": batch["text"],
                    "prompt": batch["prompt"]
                }
                outputs = model(samples)
                val_loss += outputs["loss"].item()
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        
        # Check if this is the best validation loss (ONLY for Stage 2, i.e., epoch >= 2)
        # Or if user wants to see progress in stage 1 too, we can save, but "best model selected from here [last 3]" implies final selection.
        # We will save best checkpoint updatingly.
        
        is_best = False
        if epoch >= 2:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                is_best = True
                logging.info(f"New best validation loss (Stage 2): {best_val_loss:.4f} at epoch {best_epoch}")
        
        # Save Checkpoint
        # We save every epoch for safety, but mark 'best' only in stage 2
        
        if config["model"].get("lora", True):
            # Save best checkpoint (overwrite previous best) - ONLY IN STAGE 2
            if is_best:
                model.llm_model.save_pretrained(os.path.join(output_dir, "lora_best"))
                if model.speech_proj is not None:
                    torch.save(model.speech_proj.state_dict(), os.path.join(output_dir, "proj_best.pt"))
                
                # Check if we should save audio encoder
                save_audio_encoder = (
                    not config["model"].get("freeze_audio_encoder", True)
                    or config["model"].get("use_weighted_layer_sum", False)
                )
                if save_audio_encoder:
                    torch.save(model.audio_encoder.state_dict(), os.path.join(output_dir, "audio_encoder_best.pt"))
                    
                logging.info(f"Saved best checkpoint (epoch {best_epoch}) with val_loss: {best_val_loss:.4f}")
        else:
            if is_best:
                torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_best.pt"))
                logging.info(f"Saved best checkpoint (epoch {best_epoch}) with val_loss: {best_val_loss:.4f}")
            
        logging.info(f"Saved checkpoint to {output_dir}")
    
    # Final summary
    logging.info("="*50)
    logging.info(f"Training Complete!")
    logging.info(f"Best validation loss (Stage 2): {best_val_loss:.4f} at epoch {best_epoch}")
    logging.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token")
    args = parser.parse_args()
    main(args)
