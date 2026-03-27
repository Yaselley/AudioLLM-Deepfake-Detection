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

from models_clean.audiollm_base_opensmile_selected import AudioDeepfakeLLM
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
    
    # model = AudioDeepfakeLLM(
    #     llm_path=model_cfg["llm_path"],
    #     llm_type=model_cfg.get("llm_type", "qwen2"),
    #     audio_encoder_path=model_cfg["audio_encoder_path"],
    #     audio_encoder_type=model_cfg.get("audio_encoder_type", "whisper"),
    #     audio_encoder_source=model_cfg.get("audio_encoder_source", "huggingface"),
    #     freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True),
    #     lora=model_cfg.get("lora", True),
    #     lora_rank=model_cfg.get("lora_rank", 64),
    #     lora_alpha=model_cfg.get("lora_alpha", 16),
    #     lora_dropout=model_cfg.get("lora_dropout", 0.05),
    #     special_tokens=model_cfg.get("special_tokens", []),
    #     device_8bit=0 ,
    #     speechtokenizer_num_layers=model_cfg.get("speechtokenizer_num_layers", 8),
    #     speechtokenizer_codebook_size=model_cfg.get("speechtokenizer_codebook_size", 1024),
    #     sqcodec_num_quantizers=model_cfg.get("sqcodec_num_quantizers", 4),
    #     sqcodec_num_levels=model_cfg.get("sqcodec_num_levels", 4096),
    # )
    model = AudioDeepfakeLLM(
        llm_path=model_cfg["llm_path"],
        llm_type=model_cfg.get("llm_type", "qwen2"),
        audio_encoder_path=model_cfg.get("audio_encoder_path", None),
        audio_encoder_type=model_cfg.get("audio_encoder_type", "whisper"),
        audio_encoder_source=model_cfg.get("audio_encoder_source", "huggingface"),
        freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True),
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
        speechtokenizer_return_discrete=model_cfg.get("speechtokenizer_return_discrete", True),  # ADD THIS LINE
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
    
    # Parameter Separation & Counting
    audio_enc_params = []
    adapter_params = []
    llm_params = []
    
    audio_enc_param_count = 0
    adapter_param_count = 0
    llm_param_count = 0
    layer_pooler_param_count = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "audio_encoder" in name:
            # Check if it's layer_pooler (should be trainable even if encoder is frozen)
            if "layer_pooler" in name:
                layer_pooler_param_count += param.numel()
            audio_enc_params.append(param)
            audio_enc_param_count += param.numel()
        elif "speech_proj" in name and model.speech_proj is not None:
            adapter_params.append(param)
            adapter_param_count += param.numel()
        else:
            llm_params.append(param)
            llm_param_count += param.numel()
            
    logging.info("="*30)
    logging.info(f"Trainable Parameters Info:")
    logging.info(f"  Audio Encoder Params:   {audio_enc_param_count:,} (LR: {config['train'].get('lr_audio', config['train'].get('lr', 1e-4))})")
    if layer_pooler_param_count > 0:
        logging.info(f"    └─ Layer Pooler:      {layer_pooler_param_count:,} (trainable)")
    logging.info(f"  Audio Adapter Params:   {adapter_param_count:,} (LR: {config['train'].get('lr_adapter', config['train'].get('lr', 1e-4))})")
    logging.info(f"  LLM/LoRA Params:        {llm_param_count:,} (LR: {config['train'].get('lr_lora', config['train'].get('lr', 1e-4))})")
    logging.info(f"  Total Trainable Params: {audio_enc_param_count + adapter_param_count + llm_param_count:,}")
    logging.info("="*30)

    # Optimizer with different groups
    optimizer = optim.AdamW([
        {'params': audio_enc_params, 'lr': float(config["train"].get("lr_audio", config["train"].get("lr", 1e-4)))},
        {'params': adapter_params, 'lr': float(config["train"].get("lr_adapter", config["train"].get("lr", 1e-4)))},
        {'params': llm_params, 'lr': float(config["train"].get("lr_lora", config["train"].get("lr", 1e-4)))}
    ], weight_decay=0.05)
    
    num_epochs = config["train"]["epochs"]
    
    # Training Loop
    logging.info("Starting Training...")
    global_step = 0
    
    grad_accum_steps = config["train"].get("grad_accum_steps", 1)
    
    # Track best validation loss
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
            
            # optimizer.zero_grad() # Moved to accumulation logic
            
            outputs = model(samples)
            loss = outputs["loss"] / grad_accum_steps
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            total_loss += loss.item() * grad_accum_steps
            
            if step % 1000 == 0:
                logging.info(f"Epoch {epoch+1}/{num_epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item() * grad_accum_steps:.4f}")
        
                # Print weighted layer pooling weights if enabled
                if config["model"].get("use_weighted_layer_sum", False):
                    if hasattr(model.audio_encoder, 'layer_pooler') and model.audio_encoder.layer_pooler is not None:
                        raw_weights = model.audio_encoder.layer_pooler.layer_weights.data.cpu().numpy()
                        softmax_weights = torch.softmax(model.audio_encoder.layer_pooler.layer_weights, dim=0).data.cpu().numpy()
                        logging.info(f"Epoch {epoch+1} - Weighted Layer Pooling Weights:")
                        logging.info(f"  Raw weights: {raw_weights}")
                        logging.info(f"  Softmax weights: {softmax_weights}")
                        logging.info(f"  Layer with max weight: Layer {softmax_weights.argmax()}")
            
        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1} Complete. Avg Train Loss: {avg_train_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # Validation
        model.eval()
        val_loss = 0  # Start at 0 to accumulate losses from all batches
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
        

        # Check if this is the best validation loss
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            logging.info(f"New best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        
        # Save Checkpoint (always save, but mark best)
        if config["model"].get("lora", True):
            # Save regular checkpoint
            # model.llm_model.save_pretrained(os.path.join(output_dir, f"lora_epoch_{epoch+1}"))
            
            # # Save speech projection if it exists (not for SpeechTokenizer)
            # if model.speech_proj is not None:
            #     torch.save(model.speech_proj.state_dict(), os.path.join(output_dir, f"proj_epoch_{epoch+1}.pt"))

            # # Save Audio Encoder if it has trainable params:
            # # - encoder not frozen, OR
            # # - use_weighted_layer_sum (layer_pooler is trainable even when encoder frozen)
            save_audio_encoder = (
                not config["model"].get("freeze_audio_encoder", True)
                or config["model"].get("use_weighted_layer_sum", False)
            )

            # if save_audio_encoder:
            #     torch.save(model.audio_encoder.state_dict(), os.path.join(output_dir, f"audio_encoder_epoch_{epoch+1}.pt"))

            # Save best checkpoint (overwrite previous best)
            if is_best:
                model.llm_model.save_pretrained(os.path.join(output_dir, "lora_best"))
                if model.speech_proj is not None:
                    torch.save(model.speech_proj.state_dict(), os.path.join(output_dir, "proj_best.pt"))
                if save_audio_encoder:
                    torch.save(model.audio_encoder.state_dict(), os.path.join(output_dir, "audio_encoder_best.pt"))
                logging.info(f"Saved best checkpoint (epoch {best_epoch}) with val_loss: {best_val_loss:.4f}")
        else:
            # Save regular checkpoint
            # torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            
            # Save best checkpoint
            if is_best:
                torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_best.pt"))
                logging.info(f"Saved best checkpoint (epoch {best_epoch}) with val_loss: {best_val_loss:.4f}")
            
        logging.info(f"Saved checkpoint to {output_dir}")
    
    # Final summary
    logging.info("="*50)
    logging.info(f"Training Complete!")
    logging.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    if model.speech_proj is not None:
        logging.info(f"Best checkpoint saved as: lora_best / proj_best.pt")
    else:
        logging.info(f"Best checkpoint saved as: lora_best (no projection layer for SpeechTokenizer)")
    logging.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token")
    args = parser.parse_args()
    main(args)
