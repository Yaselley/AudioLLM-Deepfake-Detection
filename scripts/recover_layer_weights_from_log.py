#!/usr/bin/env python3
"""
Recover weighted layer-pooler weights from train.log and build audio_encoder_best.pt.

Use when you trained with use_weighted_layer_sum=True but audio_encoder was not saved
(e.g. freeze_audio_encoder=True). Parses "Raw weights: [...]" from train.log, uses
config.json + checkpoint_best.pt, and saves audio_encoder_best.pt.

Usage:
  python scripts/recover_layer_weights_from_log.py --output_dir outputs/analysis_weights_nolora/w2v_weights_0.5_seed_2

Requires: config.json, train.log, checkpoint_best.pt in --output_dir (nolora runs).
Use --use_last_epoch to use last epoch's weights instead of best epoch from log.
"""
import argparse, json, os, re, sys
from typing import Dict, List, Optional
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_best_epoch(log_path):
    best = None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Best validation loss: [\d.]+ at epoch (\d+)", line)
            if m: best = int(m.group(1))
    return best

def extract_raw_weights_block(text):
    m = re.search(r"Raw weights:\s*\[([\d.\s\-eE+]+)\]", text, re.DOTALL)
    if not m: return None
    out = []
    for p in m.group(1).split():
        try: out.append(float(p))
        except ValueError: continue
    return out if out else None

def collect_epoch_weight_blocks(log_path):
    epoch_weights = {}
    current_epoch = None
    with open(log_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        me = re.search(r"Epoch (\d+) - Weighted Layer Pooling", line)
        if me: current_epoch = int(me.group(1))
        if "Raw weights:" in line and current_epoch is not None:
            buf = [line]
            while i+1 < len(lines) and not ("]" in line and line.rstrip().endswith("]")):
                i += 1
                line = lines[i]
                buf.append(line)
            w = extract_raw_weights_block("\n".join(buf))
            if w: epoch_weights[current_epoch] = w
        i += 1
    return epoch_weights

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--use_last_epoch", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.output_dir)
    config_path = os.path.join(out_dir, "config.json")
    log_path = os.path.join(out_dir, "train.log")
    ckpt_path = os.path.join(out_dir, "checkpoint_best.pt")
    # for p, name in [(config_path,"config.json"),(log_path,"train.log"),(ckpt_path,"checkpoint_best.pt")]:
    #     if not os.path.exists(p):
    #         print("Missing", name, p); sys.exit(1)
    with open(config_path) as f: config = json.load(f)
    if not config.get("model",{}).get("use_weighted_layer_sum", False):
        print("use_weighted_layer_sum False"); sys.exit(0)
    best_epoch = find_best_epoch(log_path)
    if best_epoch is None: print("No best epoch"); sys.exit(1)
    epoch_weights = collect_epoch_weight_blocks(log_path)
    if not epoch_weights: print("No Raw weights blocks"); sys.exit(1)
    epoch_to_use = max(epoch_weights.keys()) if args.use_last_epoch else best_epoch
    if epoch_to_use not in epoch_weights:
        print("No weights for epoch", epoch_to_use); sys.exit(1)
    raw_weights = epoch_weights[epoch_to_use]
    print("Extracted", len(raw_weights), "weights for epoch", epoch_to_use)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    prefix = "audio_encoder."
    audio_sd = {k[len(prefix):]: v.clone() for k,v in ckpt.items() if k.startswith(prefix)}
    if not audio_sd: print("No audio_encoder in ckpt"); sys.exit(1)
    key = "layer_pooler.layer_weights"
    if key not in audio_sd: print("No", key); sys.exit(1)
    w_tensor = torch.tensor(raw_weights, dtype=torch.float32)
    if w_tensor.shape[0] != audio_sd[key].shape[0]:
        print("Shape mismatch"); sys.exit(1)
    audio_sd[key] = w_tensor
    save_path = os.path.join(out_dir, "audio_encoder_best.pt")
    torch.save(audio_sd, save_path)
    print("Saved", save_path)

if __name__ == "__main__": main()
