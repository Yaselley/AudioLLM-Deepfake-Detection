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
            logging.FileHandler(os.path.join(output_dir, "train_stage2.log")),
            logging.StreamHandler()
        ]
    )

def main(args):
    # Authenticate with HF
    if args.hf_token:
        login(token=args.hf_token)
    else:
        login(token="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")

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update output directory to reflect Stage 2
    output_dir = config.get("output_dir", "./output")
    # You might want to append _stage2 or use the one from config if user updated it
    if args.output_dir:
        output_dir = args.output_dir
        
    setup_logging(output_dir)
    logging.info(f"Config: {config}")
    logging.info(f"Pretrained Checkpoint: {args.pretrained_checkpoint}")

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
    
    # Model Initialization (Fresh LoRA)
    logging.info("Initializing Model with Fresh LoRA...")
    model_cfg = config["model"]
    
    # Ensure LoRA is enabled in config for this stage
    model_cfg["lora"] = True 
    
    logging.info(f"DEBUG: Initializing AudioDeepfakeLLM with:")
    logging.info(f"  audio_encoder_path: {model_cfg.get('audio_encoder_path')}")
    logging.info(f"  audio_encoder_type: {model_cfg.get('audio_encoder_type')}")
    logging.info(f"  audio_encoder_source: {model_cfg.get('audio_encoder_source')}")

    model = AudioDeepfakeLLM(
        llm_path=model_cfg["llm_path"],
        llm_type=model_cfg.get("llm_type", "qwen2"),
        audio_encoder_path=model_cfg.get("audio_encoder_path", None),
        audio_encoder_type=model_cfg.get("audio_encoder_type", "whisper"),
        audio_encoder_source=model_cfg.get("audio_encoder_source", "huggingface"),
        freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True), # Initial state, will override with optimizer
        # Weighted layer pooling (Whisper / Wav2Vec2)
        use_weighted_layer_sum=model_cfg.get("use_weighted_layer_sum", False),
        weighted_layer_norm=model_cfg.get("weighted_layer_norm", "none"),
        lora=True, # Enforce LoRA
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
    
    # LOAD PRETRAINED WEIGHTS (Encoder + Adapter) from checkpoint_best.pt
    logging.info(f"Loading pretrained weights from {args.pretrained_checkpoint}...")
    
    checkpoint_path = os.path.join(args.pretrained_checkpoint, f"checkpoint_{args.checkpoint_epoch}.pt")
    
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Manual loading to avoid shape mismatch errors on specialized layers (like layer_norm in Fairseq vs HF wrappers)
        # and to ignore keys that don't match.
        model_state_dict = model.state_dict()
        pretrained_dict = {}
        shape_mismatch_keys = []
        
        for k, v in checkpoint.items():
            candidates = [k]
            
            # 1. Handle double encoder nesting (remove one)
            if "audio_encoder.encoder.encoder." in k:
                candidates.append(k.replace("audio_encoder.encoder.encoder.", "audio_encoder.encoder."))
            
            # 2. Handle missing encoder nesting (add one)
            if "audio_encoder.encoder." in k and "audio_encoder.encoder.encoder." not in k:
                candidates.append(k.replace("audio_encoder.encoder.", "audio_encoder.encoder.encoder."))
            
            # 3. Handle missing encoder wrapper (e.g. EAT/Whisper checkpoint having .layers directly)
            if "audio_encoder.layers." in k:
                candidates.append(k.replace("audio_encoder.layers.", "audio_encoder.encoder.layers."))

            # 4. Handle extra encoder wrapper (reverse of 3)
            if "audio_encoder.encoder.layers." in k:
                candidates.append(k.replace("audio_encoder.encoder.layers.", "audio_encoder.layers."))

            # 5. Handle LLM wrapping
            if "llm_model.base_model.model.model." in k:
                candidates.append(k.replace("llm_model.base_model.model.model.", "llm_model.model."))

            match_found = False
            for cand in candidates:
                if cand in model_state_dict:
                    if v.shape == model_state_dict[cand].shape:
                        pretrained_dict[cand] = v
                        match_found = True
                        break # Stop at first valid match
                    else:
                        # Log mismatch but keep looking for other candidates (unlikely to have multiple candidates with different shapes, but safe)
                        logging.warning(f"Shape mismatch for candidate {cand} (from {k}): checkpoint {v.shape} vs model {model_state_dict[cand].shape}")

            if not match_found and "audio_encoder" in k:
                 # Optional: log if an audio key was dropped entirely
                 # logging.warning(f"Dropped audio key: {k}")
                 pass

        # Load the filtered dictionary
        msg = model.load_state_dict(pretrained_dict, strict=False)
        logging.info(f"Checkpoint loaded (partial). Shape mismatches ignored: {len(shape_mismatch_keys)}")
        if len(shape_mismatch_keys) > 0:
            logging.info(f"First 5 mismatched keys: {shape_mismatch_keys[:5]}")

        # Verify if speech_proj and audio_encoder were likely loaded
        if any("speech_proj" in k for k in pretrained_dict.keys()):
            logging.info("Confirmed: speech_proj weights loaded from checkpoint.")
        else:
            logging.warning("WARNING: speech_proj weights NOT loaded from checkpoint!")
            
        if any("audio_encoder" in k for k in pretrained_dict.keys()):
            logging.info("Confirmed: audio_encoder weights loaded from checkpoint.")
        else:
            logging.warning("WARNING: audio_encoder weights NOT loaded from checkpoint!")
            
    else:
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        # Fallback logic or exit? For now, we assume user provided correct path.
        # We can try falling back to the separated files just in case
        logging.info("Attempting fallback to separate component files (proj_X.pt, audio_encoder_X.pt)...")
        
        # 1. Projector
        if model.speech_proj is not None:
            proj_path = os.path.join(args.pretrained_checkpoint, f"proj_{args.checkpoint_epoch}.pt")
            if os.path.exists(proj_path):
                logging.info(f"Loading projector weights from {proj_path}")
                model.speech_proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
            else:
                logging.warning(f"Projector weights not found at {proj_path}!")

        # 2. Audio Encoder
        audio_enc_path = os.path.join(args.pretrained_checkpoint, f"audio_encoder_{args.checkpoint_epoch}.pt")
        if os.path.exists(audio_enc_path):
            logging.info(f"Loading audio encoder weights from {audio_enc_path}")
            model.audio_encoder.load_state_dict(torch.load(audio_enc_path, map_location="cpu"))
        else:
            logging.info(f"No custom audio encoder weights found at {audio_enc_path}.")

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
    
    # Configure Optimizer for Stage 2 (LoRA + Encoder + Adapter)
    # Enable gradients for all we want to tune
    
    freeze_audio_encoder = config["model"].get("freeze_audio_encoder", False) 
    
    # Collect params
    audio_enc_params = []
    adapter_params = []
    llm_params = []
    
    audio_enc_param_count = 0
    adapter_param_count = 0
    llm_param_count = 0
    
    for name, param in model.named_parameters():
        # Default logic for Stage 2
        if "audio_encoder" in name:
            if "layer_pooler" in name:
                 param.requires_grad = True
            else:
                 param.requires_grad = not freeze_audio_encoder
            
            if param.requires_grad:
                audio_enc_params.append(param)
                audio_enc_param_count += param.numel()
                
        elif "speech_proj" in name and model.speech_proj is not None:
            param.requires_grad = True
            adapter_params.append(param)
            adapter_param_count += param.numel()
            
        else:
            # LLM / LoRA
            # In Stage 2, we want to train LoRA. 
            # Check if it's a LoRA parameter
            if "lora_" in name or "modules_to_save" in name or "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
                llm_params.append(param)
                llm_param_count += param.numel()
            else:
                param.requires_grad = False

    logging.info("="*30)
    logging.info(f"Stage 2 Trainable Parameters:")
    logging.info(f"  Audio Encoder: {audio_enc_param_count:,}")
    logging.info(f"  Audio Adapter: {adapter_param_count:,}")
    logging.info(f"  LLM (LoRA):    {llm_param_count:,}")
    logging.info(f"  Total:         {audio_enc_param_count + adapter_param_count + llm_param_count:,}")
    logging.info("="*30)

    # Optimizer
    params_group = []
    if audio_enc_params:
        params_group.append({'params': audio_enc_params, 'lr': float(config["train"].get("lr_audio", config["train"].get("lr", 1e-4)))})
    if adapter_params:
        params_group.append({'params': adapter_params, 'lr': float(config["train"].get("lr_adapter", config["train"].get("lr", 1e-4)))})
    if llm_params:
        params_group.append({'params': llm_params, 'lr': float(config["train"].get("lr_lora", config["train"].get("lr", 1e-4)))})
        
    optimizer = optim.AdamW(params_group, weight_decay=0.05)
    
    num_epochs = config["train"].get("epochs_stage2", 3) # Use specific epoch count for stage 2, default 3
    grad_accum_steps = config["train"].get("grad_accum_steps", 1)
    
    # Training Loop
    logging.info(f"Starting Stage 2 Training for {num_epochs} epochs...")
    global_step = 0
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
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
        logging.info(f"Epoch {epoch+1} Complete. Avg Train Loss: {avg_train_loss:.4f} | Time: {time.time() - start_time:.2f}s")
        
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
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            
            model.llm_model.save_pretrained(os.path.join(output_dir, "lora_best"))
            if model.speech_proj is not None:
                torch.save(model.speech_proj.state_dict(), os.path.join(output_dir, "proj_best.pt"))
            
            if not freeze_audio_encoder or config["model"].get("use_weighted_layer_sum", False):
                torch.save(model.audio_encoder.state_dict(), os.path.join(output_dir, "audio_encoder_best.pt"))
                
            logging.info(f"New best model saved at epoch {best_epoch} with val_loss: {best_val_loss:.4f}")

    logging.info("Stage 2 Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True, help="Path to checkpoint with trained encoder/adapter")
    parser.add_argument("--checkpoint_epoch", type=str, default="best", help="Epoch suffix for checkpoint files (e.g., 'best', '2', '5')")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory override")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token")
    args = parser.parse_args()
    main(args)