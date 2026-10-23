#!/usr/bin/env python3
"""Stage-2 training from a previously trained SFT or GRPO checkpoint."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from train_common import (
    build_dataloaders,
    build_model,
    load_config,
    maybe_hf_login,
    run_training_loop,
    save_config,
    set_seed,
    setup_logging,
)


def _load_pretrained(
    model,
    checkpoint_dir: str,
    checkpoint_epoch: str,
) -> None:
    ckpt = Path(checkpoint_dir) / f"checkpoint_{checkpoint_epoch}.pt"
    if ckpt.is_file():
        logging.info("Loading full checkpoint %s", ckpt)
        state = torch.load(ckpt, map_location="cpu")
        model_state = model.state_dict()
        loaded = {}
        for key, value in state.items():
            if key in model_state and value.shape == model_state[key].shape:
                loaded[key] = value
        model.load_state_dict(loaded, strict=False)
        logging.info("Loaded %s matching tensors from checkpoint", len(loaded))
        return

    proj = Path(checkpoint_dir) / f"proj_{checkpoint_epoch}.pt"
    enc = Path(checkpoint_dir) / f"audio_encoder_{checkpoint_epoch}.pt"
    if model.speech_proj is not None and proj.is_file():
        logging.info("Loading projector from %s", proj)
        model.speech_proj.load_state_dict(torch.load(proj, map_location="cpu"))
    if enc.is_file():
        logging.info("Loading audio encoder from %s", enc)
        model.audio_encoder.load_state_dict(torch.load(enc, map_location="cpu"))
    if not proj.is_file() and not enc.is_file():
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
    lora_dir = Path(checkpoint_dir) / (
        f"lora_{checkpoint_epoch}"
        if checkpoint_epoch != "best"
        else "lora_best"
    )

    if lora_dir.is_dir():
        logging.info("Loading LoRA adapter from %s", lora_dir)
        model.llm_model.load_adapter(
            str(lora_dir),
            adapter_name="default",
            is_trainable=True,
        )


def _configure_stage2_params(model, config: dict) -> list:
    """Enable gradients for LoRA, adapter, and optionally the audio encoder."""
    freeze_enc = config["model"].get("freeze_audio_encoder", False)
    groups = {"audio": [], "adapter": [], "lora": []}

    if config["model"].get("mlp_adaptor_only", False):
        for name, param in model.named_parameters():
            param.requires_grad = "mlp_adaptor" in name
            if param.requires_grad:
                groups["adapter"].append(param)
        logging.info(
            "MLP-adaptor-only stage-2 training — trainable params: %s",
            sum(param.numel() for param in groups["adapter"]),
        )
        train_cfg = config["train"]
        return [{
            "params": groups["adapter"],
            "lr": float(train_cfg.get("lr_adapter", train_cfg.get("lr", 1e-4))),
        }]

    for name, param in model.named_parameters():
        if "audio_encoder" in name:
            param.requires_grad = ("layer_pooler" in name) or not freeze_enc
            if param.requires_grad:
                groups["audio"].append(param)
        elif "speech_proj" in name and model.speech_proj is not None:
            param.requires_grad = True
            groups["adapter"].append(param)
        elif "lora_" in name or "modules_to_save" in name:
            param.requires_grad = True
            groups["lora"].append(param)
        else:
            param.requires_grad = False

    train_cfg = config["train"]
    optim_groups = []
    for key, lr_key in (("audio", "lr_audio"), ("adapter", "lr_adapter"), ("lora", "lr_lora")):
        if groups[key]:
            optim_groups.append(
                {
                    "params": groups[key],
                    "lr": float(train_cfg.get(lr_key, train_cfg.get("lr", 1e-4))),
                }
            )
    logging.info(
        "Stage-2 trainable params — encoder: %s, adapter: %s, lora: %s",
        sum(p.numel() for p in groups["audio"]),
        sum(p.numel() for p in groups["adapter"]),
        sum(p.numel() for p in groups["lora"]),
    )
    return optim_groups


def main(
    config_path: str,
    pretrained_checkpoint: str,
    checkpoint_epoch: str = "best",
    output_dir: str | None = None,
    hf_token: str | None = None,
) -> None:
    maybe_hf_login(hf_token)
    config = load_config(config_path)
    # config["model"]["lora"] = True
    if output_dir:
        config["output_dir"] = output_dir
    output_dir = config.get("output_dir", "./output")

    setup_logging(output_dir, log_name="train_stage2.log")
    save_config(config, output_dir)
    logging.info("Pretrained checkpoint: %s", pretrained_checkpoint)

    set_seed(config["train"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config["model"])
    _load_pretrained(model, pretrained_checkpoint, checkpoint_epoch)
    model.to(device)

    train_loader, val_loader = build_dataloaders(config)
    optimizer = torch.optim.AdamW(_configure_stage2_params(model, config), weight_decay=0.05)
    epochs = config["train"].get("epochs_stage2", config["train"].get("epochs", 5))

    run_training_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        config,
        output_dir,
        device,
        num_epochs=epochs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--checkpoint_epoch", default="best")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()
    main(
        args.config,
        args.pretrained_checkpoint,
        args.checkpoint_epoch,
        args.output_dir,
        args.hf_token,
    )