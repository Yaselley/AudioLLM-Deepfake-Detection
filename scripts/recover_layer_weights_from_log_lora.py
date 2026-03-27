#!/usr/bin/env python3
"""
Recover weighted layer-pooler weights from train.log for LoRA runs and build audio_encoder_best.pt.

Use when you trained with use_weighted_layer_sum=True and lora=True; the run saves
lora_best/ + proj_best.pt but not a full checkpoint, so audio_encoder (layer_pooler)
was never saved. This script parses "Raw weights: [...]" from train.log and saves
audio_encoder_best.pt containing only layer_pooler.layer_weights.

Usage:
  python scripts/recover_layer_weights_from_log_lora.py --output_dir outputs/analysis_weights/w2v_weights_0.5_seed_2

Requires: config.json, train.log in --output_dir (LoRA runs: no checkpoint_best.pt).
Use --use_last_epoch to use last epoch's weights instead of best epoch from log.
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_best_epoch(log_path: str) -> Optional[int]:
    """Find best epoch from 'Best validation loss: X at epoch N' or 'New best validation loss: X at epoch N'."""
    best = None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Best validation loss: [\d.]+ at epoch (\d+)", line)
            if m:
                best = int(m.group(1))
            m2 = re.search(r"New best validation loss: [\d.]+ at epoch (\d+)", line)
            if m2:
                best = int(m2.group(1))
    return best


def extract_raw_weights_block(text: str) -> Optional[List[float]]:
    m = re.search(r"Raw weights:\s*\[([\d.\s\-eE+]+)\]", text, re.DOTALL)
    if not m:
        return None
    out = []
    for p in m.group(1).split():
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out if out else None


def collect_epoch_weight_blocks(log_path: str) -> Dict[int, List[float]]:
    """Collect last 'Raw weights' block per epoch (matches 'Epoch N - Weighted Layer Pooling' or '...Weights')."""
    epoch_weights: Dict[int, List[float]] = {}
    current_epoch = None
    with open(log_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        me = re.search(r"Epoch (\d+) - Weighted Layer Pooling", line)
        if me:
            current_epoch = int(me.group(1))
        if "Raw weights:" in line and current_epoch is not None:
            buf = [line]
            while i + 1 < len(lines) and not (
                "]" in line and line.rstrip().endswith("]")
            ):
                i += 1
                line = lines[i]
                buf.append(line)
            w = extract_raw_weights_block("\n".join(buf))
            if w:
                epoch_weights[current_epoch] = w
        i += 1
    return epoch_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True, help="Run output dir (e.g. outputs/analysis_weights/w2v_weights_0.5_seed_2)")
    ap.add_argument("--use_last_epoch", action="store_true", help="Use last epoch's weights instead of best")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    config_path = os.path.join(out_dir, "config.json")
    log_path = os.path.join(out_dir, "train.log")

    if not os.path.exists(config_path):
        print("Missing config.json at", config_path)
        sys.exit(1)
    if not os.path.exists(log_path):
        print("Missing train.log at", log_path)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    if not config.get("model", {}).get("use_weighted_layer_sum", False):
        print("use_weighted_layer_sum is False in config; nothing to recover.")
        sys.exit(0)

    best_epoch = find_best_epoch(log_path)
    if best_epoch is None:
        print("No best epoch found in train.log")
        sys.exit(1)

    epoch_weights = collect_epoch_weight_blocks(log_path)
    if not epoch_weights:
        print("No 'Raw weights' blocks found in train.log")
        sys.exit(1)

    epoch_to_use = max(epoch_weights.keys()) if args.use_last_epoch else best_epoch
    if epoch_to_use not in epoch_weights:
        print("No weights for epoch", epoch_to_use)
        sys.exit(1)

    raw_weights = epoch_weights[epoch_to_use]
    print("Extracted", len(raw_weights), "layer weights for epoch", epoch_to_use)

    # LoRA runs: no checkpoint_best.pt; build audio_encoder state dict from log only.
    w_tensor = torch.tensor(raw_weights, dtype=torch.float32)
    audio_sd: Dict[str, torch.Tensor] = {
        "layer_pooler.layer_weights": w_tensor,
    }


    save_path = os.path.join(out_dir, "layer_best.pt")
    torch.save(audio_sd, save_path)
    print("Saved", save_path)


if __name__ == "__main__":
    main()
