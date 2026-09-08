#!/usr/bin/env python3
"""Train AudioLLM with GRPO reasoning-reward fine-tuning.

Same shape as train_baseline.py — only the training loop import changes
(run_training_loop -> run_training_loop_grpo). Add a `grpo:` block under
`train:` in your YAML config, e.g.:

    train:
      epochs: 6
      batch_size: 4
      grad_accum_steps: 4
      lr: 1e-5
      grpo:
        enabled: true
        start_epoch: 0
        num_rollouts: 4
        max_new_tokens: 128
        temperature: 0.8
        top_p: 0.9
        repetition_penalty: 1.1
        clip_epsilon: 0.2
        kl_beta: 0.05
        log_freq: 50
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path 
import torch

from grpo import run_training_loop_grpo
from train_common import (
    build_dataloaders,
    build_model,
    build_optimizer,
    load_config,
    maybe_hf_login,
    save_config,
    set_seed,
    setup_logging,
    init_distributed,
    distributed_barrier,
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





def main(config_path: str, pretrained_checkpoint, checkpoint_epoch="best",  hf_token: str | None = None) -> None:
    device, is_main_process = init_distributed()
    maybe_hf_login(hf_token)
    config = load_config(config_path)
    output_dir = config.get("output_dir", "./output")
    rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
    setup_logging(output_dir, log_name="train.log" if is_main_process else f"train_rank{rank}.log")
    if is_main_process:
        save_config(config, output_dir)
    distributed_barrier()
    logging.info("Config: %s", config)

    set_seed(config["train"].get("seed", 42))
    if is_main_process:
        logging.info("Device: %s", device)

    model = build_model(config["model"]).to(device)
    _load_pretrained(model, pretrained_checkpoint, checkpoint_epoch)
    train_loader, val_loader = build_dataloaders(config)
    optimizer = build_optimizer(model, config)
    run_training_loop_grpo(
        model,
        train_loader,
        val_loader,
        optimizer,
        config,
        output_dir,
        device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--checkpoint_epoch", default="best")

    parser.add_argument("--hf_token", default=None, help="HF token (or set HF_TOKEN)")
    args = parser.parse_args()
    main(args.config, args.pretrained_checkpoint, args.checkpoint_epoch, args.hf_token)