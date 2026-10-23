#!/usr/bin/env python3
"""Train the AudioLLM (XLS-R + Qwen3.5) with supervised fine-tuning."""

from __future__ import annotations

import argparse
import logging
import os

import torch

from train_common import (
    build_dataloaders,
    build_model,
    build_optimizer,
    load_config,
    maybe_hf_login,
    run_training_loop,
    save_config,
    set_seed,
    setup_logging,
)


def main(config_path: str, hf_token: str | None = None) -> None:
    maybe_hf_login(hf_token)
    config = load_config(config_path)
    output_dir = config.get("output_dir", "./output")
    setup_logging(output_dir)
    save_config(config, output_dir)
    logging.info("Config: %s", config)

    set_seed(config["train"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    model = build_model(config["model"]).to(device)
    train_loader, val_loader = build_dataloaders(config)
    optimizer = build_optimizer(model, config)
    run_training_loop(
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
    parser.add_argument("--hf_token", default=None, help="HF token (or set HF_TOKEN)")
    args = parser.parse_args()
    main(args.config, args.hf_token)
