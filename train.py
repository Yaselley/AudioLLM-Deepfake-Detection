import os
import argparse
import yaml
import logging
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import time

from models.audio_llm import AudioDeepfakeLLM
from dataset import AudioDeepfakeDataset

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
    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = config.get("output_dir", "./output")
    setup_logging(output_dir)
    logging.info(f"Config: {config}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Model
    logging.info("Initializing Model...")
    model_cfg = config["model"]
    
    model = AudioDeepfakeLLM(
        llm_path=model_cfg["llm_path"],
        llm_type=model_cfg.get("llm_type", "qwen2"),
        audio_encoder_path=model_cfg["audio_encoder_path"],
        audio_encoder_type=model_cfg.get("audio_encoder_type", "whisper"),
        freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True),
        lora=model_cfg.get("lora", True),
        lora_rank=model_cfg.get("lora_rank", 8),
        device_8bit=0 
    )
    model.to(device)
    
    # Dataset
    logging.info("Loading Datasets...")
    # Dataset expects raw audio paths. Feature extraction (spectrogram) happens in Model if needed.
    train_dataset = AudioDeepfakeDataset(
        config["data"]["train_path"]
    )
    val_dataset = AudioDeepfakeDataset(
        config["data"]["val_path"]
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
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["train"]["lr"]),
        weight_decay=0.05
    )
    
    num_epochs = config["train"]["epochs"]
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * num_training_steps), 
        num_training_steps=num_training_steps
    )
    
    # Training Loop
    logging.info("Starting Training...")
    global_step = 0
    
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
            
            optimizer.zero_grad()
            
            outputs = model(samples)
            loss = outputs["loss"]
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            global_step += 1
            
            if step % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{num_epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
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
        
        # Save Checkpoint
        ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        # Save PEFT model weights or full model
        if config["model"].get("lora", True):
            model.llm_model.save_pretrained(os.path.join(output_dir, f"lora_epoch_{epoch+1}"))
            torch.save(model.speech_proj.state_dict(), os.path.join(output_dir, f"proj_epoch_{epoch+1}.pt"))
        else:
            torch.save(model.state_dict(), ckpt_path)
            
        logging.info(f"Saved checkpoint to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    main(args)
