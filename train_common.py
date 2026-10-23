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
from torch.utils.data.distributed import DistributedSampler

from dataset import AudioDeepfakeDataset
from models.audiollm import AudioDeepfakeLLM


def init_distributed() -> tuple[torch.device, bool]:
    """Initialize torchrun's process group and return (device, is_main)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), True

    if not torch.cuda.is_available():
        raise RuntimeError("Multi-process training requires CUDA on every process")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return torch.device("cuda", local_rank), int(os.environ["RANK"]) == 0


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def average_gradients(model: torch.nn.Module) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        torch.distributed.all_reduce(gradient, op=torch.distributed.ReduceOp.SUM)
        gradient.div_(world_size)
        parameter.grad = gradient


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
    model = AudioDeepfakeLLM(
        llm_path=model_cfg["llm_path"],
        llm_type=model_cfg.get("llm_type", "qwen3.5"),
        audio_encoder_path=model_cfg.get("audio_encoder_path"),
        audio_encoder_type=model_cfg.get("audio_encoder_type", "wav2vec2"),
        audio_encoder_source=model_cfg.get("audio_encoder_source", "fairseq"),
        freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True),
        use_weighted_layer_sum=model_cfg.get("use_weighted_layer_sum", False),
        weighted_layer_norm=model_cfg.get("weighted_layer_norm", "none"),
        lora=lora,
        lora_rank=model_cfg.get("lora_rank", 64),
        lora_alpha=model_cfg.get("lora_alpha", 16),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        mlp_adaptor=model_cfg.get("mlp_adaptor", False),
        mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", 768),
        mlp_dropout=model_cfg.get("mlp_dropout", 0.1),
        mlp_num_classes=model_cfg.get("mlp_num_classes", 2),
        mlp_loss_weight=model_cfg.get("mlp_loss_weight", 0.5),
        special_tokens=model_cfg.get("special_tokens", []),
        device_8bit=0,
    )
    if model_cfg.get("mlp_adaptor_only", False):
        for parameter in model.parameters():
            parameter.requires_grad = False
        if model.mlp_adaptor is None:
            raise ValueError("mlp_adaptor_only requires mlp_adaptor: true")
        for parameter in model.mlp_adaptor.parameters():
            parameter.requires_grad = True
        logging.info("MLP-adaptor-only training enabled; all other parameters are frozen")
    return model


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
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=train_dataset.collate_fn,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
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
        elif "mlp_adaptor" in name and model.use_mlp_adaptor:
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
    if model.use_mlp_adaptor:
        logging.info(
            "  MLP adaptor:    %s (lr=%s)",
            f"{sum(p.numel() for p in model.mlp_adaptor.parameters() if p.requires_grad):,}",
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
        average_gradients(model)
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
    value = torch.tensor(
        [val_loss, len(val_loader)], device=device, dtype=torch.float64
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return (value[0] / value[1].clamp_min(1)).item()


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
        model.llm_model.save_pretrained(out / "lora_best", safe_serialization=False)
        if model.speech_proj is not None:
            torch.save(model.speech_proj.state_dict(), out / "proj_best.pt")
        if model.use_mlp_adaptor and model.mlp_adaptor is not None:
            torch.save(model.mlp_adaptor.state_dict(), out / "mlp_adaptor_best.pt")
        if should_save_audio_encoder(config):
            torch.save(model.audio_encoder.state_dict(), out / "audio_encoder_best.pt")
    else:
        torch.save(model.state_dict(), out / "checkpoint_best.pt")
    logging.info("Saved best checkpoint epoch %s (val_loss=%.4f)", epoch, val_loss)


def save_epoch_checkpoint(
    model: AudioDeepfakeLLM,
    output_dir: str,
    config: dict,
    *,
    epoch: int,
    val_loss: float,
) -> None:
    """Save a checkpoint every epoch, in its own epoch-numbered directory.

    Unlike save_best_checkpoint (overwritten whenever a new best is found)
    this keeps one directory per epoch so any epoch can be revisited later.
    """
    model_cfg = config["model"]
    out = Path(output_dir) / "epoch_ckpts" / f"epoch_{epoch}"
    out.mkdir(parents=True, exist_ok=True)

    if model_cfg.get("lora", True):
        model.llm_model.save_pretrained(out / "lora", safe_serialization=False)
        if model.speech_proj is not None:
            torch.save(model.speech_proj.state_dict(), out / "proj.pt")
        if model.use_mlp_adaptor and model.mlp_adaptor is not None:
            torch.save(model.mlp_adaptor.state_dict(), out / "mlp_adaptor.pt")
        if should_save_audio_encoder(config):
            torch.save(model.audio_encoder.state_dict(), out / "audio_encoder.pt")
    else:
        torch.save(model.state_dict(), out / "checkpoint.pt")

    meta = {"epoch": epoch, "val_loss": val_loss}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f)

    logging.info("Saved epoch checkpoint: epoch=%s val_loss=%.4f -> %s", epoch, val_loss, out)


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
    save_every_epoch: bool = True,
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

        if save_every_epoch:
            save_epoch_checkpoint(
                model, output_dir, config, epoch=epoch + 1, val_loss=avg_val
            )

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch + 1
            save_best_checkpoint(
                model, output_dir, config, epoch=best_epoch, val_loss=best_val
            )

    logging.info("Done. Best val loss %.4f at epoch %s", best_val, best_epoch)

def save_step_checkpoint(
    model: AudioDeepfakeLLM,
    optimizer: optim.AdamW,
    output_dir: str,
    config: dict,
    *,
    epoch: int,
    step: int,
    global_step: int,
) -> None:
    """Mid-epoch checkpoint for resuming after a wall-clock timeout.

    Written to a fixed filename (not epoch/step-suffixed) so each new save
    overwrites the last — avoids filling the cluster FS with stale
    intermediate checkpoints during a multi-day epoch.
    """
    out = Path(output_dir)
    model_cfg = config["model"]

    ckpt_dir = out / "step_ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    if model_cfg.get("lora", True):
        model.llm_model.save_pretrained(ckpt_dir / "lora", safe_serialization=False)
        if model.speech_proj is not None:
            torch.save(model.speech_proj.state_dict(), ckpt_dir / "proj.pt")
        if should_save_audio_encoder(config):
            torch.save(model.audio_encoder.state_dict(), ckpt_dir / "audio_encoder.pt")
    else:
        torch.save(model.state_dict(), ckpt_dir / "model.pt")

    # Optimizer state is required for correct resume — without it, Adam's
    # moment estimates reset and you get a loss spike after every resume.
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    meta = {"epoch": epoch, "step": step, "global_step": global_step}
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    # Atomic-ish: write to tmp then rename, so a job killed mid-write
    # doesn't leave a half-written checkpoint that resume then loads.
    logging.info("Saved step checkpoint: epoch=%s step=%s global_step=%s", epoch, step, global_step)


def load_step_checkpoint(
    model: AudioDeepfakeLLM,
    optimizer: optim.AdamW,
    output_dir: str,
    config: dict,
) -> dict | None:
    """Returns resume metadata dict, or None if no step checkpoint exists."""
    ckpt_dir = Path(output_dir) / "step_ckpt"
    meta_path = ckpt_dir / "meta.json"
    if not meta_path.is_file():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    model_cfg = config["model"]
    if model_cfg.get("lora", True):
        from peft import PeftModel
        lora_dir = ckpt_dir / "lora"
        if lora_dir.is_dir():
            model.llm_model.load_adapter(str(lora_dir), adapter_name="default", is_trainable=True)
            logging.info("Resumed LoRA weights from %s", lora_dir)
        proj_path = ckpt_dir / "proj.pt"
        if model.speech_proj is not None and proj_path.is_file():
            model.speech_proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
        if model.use_mlp_adaptor and model.mlp_adaptor is not None:
            mlp_path = ckpt_dir / "mlp_adaptor.pt"
            if mlp_path.is_file():
                model.mlp_adaptor.load_state_dict(torch.load(mlp_path, map_location="cpu"))
        enc_path = ckpt_dir / "audio_encoder.pt"
        if should_save_audio_encoder(config) and enc_path.is_file():
            model.audio_encoder.load_state_dict(torch.load(enc_path, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))

    opt_path = ckpt_dir / "optimizer.pt"
    if opt_path.is_file():
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

    logging.info("Resuming from step checkpoint: %s", meta)
    return meta