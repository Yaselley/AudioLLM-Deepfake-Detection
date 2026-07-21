"""Shared helpers for AudioLLM training scripts."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from huggingface_hub import login
from torch.utils.data import DataLoader

from dataset import AudioDeepfakeDataset
from models.audiollm_base_opensmile import AudioDeepfakeLLM


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def maybe_hf_login(hf_token: str | None = None) -> None:
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)


def setup_logging(output_dir: str, log_name: str = "train.log") -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, log_name)),
            logging.StreamHandler(),
        ],
        force=True,
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict, output_dir: str) -> None:
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def build_model(model_cfg: dict, *, force_lora: bool | None = None) -> AudioDeepfakeLLM:
    lora = force_lora if force_lora is not None else model_cfg.get("lora", True)
    return AudioDeepfakeLLM(
        llm_path=model_cfg["llm_path"],
        llm_type=model_cfg.get("llm_type", "qwen2"),
        audio_encoder_path=model_cfg.get("audio_encoder_path"),
        audio_encoder_type=model_cfg.get("audio_encoder_type", "whisper"),
        audio_encoder_source=model_cfg.get("audio_encoder_source", "huggingface"),
        freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True),
        use_weighted_layer_sum=model_cfg.get("use_weighted_layer_sum", False),
        weighted_layer_norm=model_cfg.get("weighted_layer_norm", "none"),
        lora=lora,
        lora_rank=model_cfg.get("lora_rank", 64),
        lora_alpha=model_cfg.get("lora_alpha", 16),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        special_tokens=model_cfg.get("special_tokens", []),
        device_8bit=0,
        use_opensmile=model_cfg.get("use_opensmile", False),
        opensmile_only=model_cfg.get("opensmile_only", False),
    )


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    train_dataset = AudioDeepfakeDataset(
        config["data"]["train_path"],
        prompt_path=config["data"].get("prompt_path"),
    )
    val_dataset = AudioDeepfakeDataset(
        config["data"]["val_path"],
        prompt_path=config["data"].get("prompt_path"),
    )
    common = dict(
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"].get("num_workers", 4),
        pin_memory=True,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        **common,
    )
    return train_loader, val_loader


def collect_param_groups(model: AudioDeepfakeLLM, config: dict) -> tuple[list, list, list]:
    """Return (audio_enc_params, adapter_params, llm_params) for optimizer."""
    audio_enc_params, adapter_params, llm_params = [], [], []
    layer_pooler_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "audio_encoder" in name:
            if "layer_pooler" in name:
                layer_pooler_count += param.numel()
            audio_enc_params.append(param)
        elif "speech_proj" in name and model.speech_proj is not None:
            adapter_params.append(param)
        else:
            llm_params.append(param)

    train_cfg = config["train"]
    logging.info("=" * 30)
    logging.info("Trainable parameters:")
    logging.info(
        "  Audio encoder: %s (lr=%s)",
        f"{sum(p.numel() for p in audio_enc_params):,}",
        train_cfg.get("lr_audio", train_cfg.get("lr", 1e-4)),
    )
    if layer_pooler_count:
        logging.info("    layer pooler: %s", f"{layer_pooler_count:,}")
    logging.info(
        "  Speech adapter: %s (lr=%s)",
        f"{sum(p.numel() for p in adapter_params):,}",
        train_cfg.get("lr_adapter", train_cfg.get("lr", 1e-4)),
    )
    logging.info(
        "  LLM / LoRA:     %s (lr=%s)",
        f"{sum(p.numel() for p in llm_params):,}",
        train_cfg.get("lr_lora", train_cfg.get("lr", 1e-4)),
    )
    logging.info(
        "  Total:          %s",
        f"{sum(p.numel() for p in audio_enc_params + adapter_params + llm_params):,}",
    )
    logging.info("=" * 30)
    return audio_enc_params, adapter_params, llm_params


def build_optimizer(model: AudioDeepfakeLLM, config: dict) -> optim.AdamW:
    audio_enc_params, adapter_params, llm_params = collect_param_groups(model, config)
    groups = []
    train_cfg = config["train"]
    for params, lr_key in (
        (audio_enc_params, "lr_audio"),
        (adapter_params, "lr_adapter"),
        (llm_params, "lr_lora"),
    ):
        if params:
            groups.append(
                {
                    "params": params,
                    "lr": float(train_cfg.get(lr_key, train_cfg.get("lr", 1e-4))),
                }
            )
    return optim.AdamW(groups, weight_decay=0.05)


def batch_to_samples(batch: dict, device: torch.device) -> dict:
    raw_wav = batch["raw_wav"].to(device)
    audio_attention_mask = batch.get("audio_attention_mask")
    if audio_attention_mask is not None:
        audio_attention_mask = audio_attention_mask.to(device)
    return {
        "raw_wav": raw_wav,
        "audio_attention_mask": audio_attention_mask,
        "text": batch["text"],
        "prompt": batch["prompt"],
    }


def train_one_epoch(
    model: AudioDeepfakeLLM,
    train_loader: DataLoader,
    optimizer: optim.AdamW,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    grad_accum_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader):
        outputs = model(batch_to_samples(batch, device))
        loss = outputs["loss"] / grad_accum_steps
        loss.backward()
        total_loss += outputs["loss"].item()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 1000 == 0:
            logging.info(
                "Epoch %s/%s | step %s/%s | loss %.4f",
                epoch + 1,
                num_epochs,
                step,
                len(train_loader),
                outputs["loss"].item(),
            )

    logging.info(
        "Epoch %s train loss %.4f (%.1fs)",
        epoch + 1,
        total_loss / len(train_loader),
        time.time() - start,
    )
    return total_loss / len(train_loader)


def validate(model: AudioDeepfakeLLM, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch_to_samples(batch, device))
            val_loss += outputs["loss"].item()
    return val_loss / len(val_loader)


def should_save_audio_encoder(config: dict) -> bool:
    model_cfg = config["model"]
    return not model_cfg.get("freeze_audio_encoder", True) or model_cfg.get(
        "use_weighted_layer_sum", False
    )


def save_best_checkpoint(
    model: AudioDeepfakeLLM,
    output_dir: str,
    config: dict,
    *,
    epoch: int,
    val_loss: float,
) -> None:
    model_cfg = config["model"]
    out = Path(output_dir)
    if model_cfg.get("lora", True):
        model.llm_model.save_pretrained(out / "lora_best")
        if model.speech_proj is not None:
            torch.save(model.speech_proj.state_dict(), out / "proj_best.pt")
        if should_save_audio_encoder(config):
            torch.save(model.audio_encoder.state_dict(), out / "audio_encoder_best.pt")
    else:
        torch.save(model.state_dict(), out / "checkpoint_best.pt")
    logging.info("Saved best checkpoint epoch %s (val_loss=%.4f)", epoch, val_loss)


def run_training_loop(
    model: AudioDeepfakeLLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.AdamW,
    config: dict,
    output_dir: str,
    device: torch.device,
    *,
    num_epochs: int | None = None,
) -> None:
    num_epochs = num_epochs or config["train"]["epochs"]
    grad_accum = config["train"].get("grad_accum_steps", 1)
    best_val = float("inf")
    best_epoch = 0

    for epoch in range(num_epochs):
        train_one_epoch(
            model, train_loader, optimizer, device, epoch, num_epochs, grad_accum
        )
        avg_val = validate(model, val_loader, device)
        logging.info("Epoch %s val loss %.4f", epoch + 1, avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch + 1
            save_best_checkpoint(
                model, output_dir, config, epoch=best_epoch, val_loss=best_val
            )

    logging.info("Done. Best val loss %.4f at epoch %s", best_val, best_epoch)
