#!/usr/bin/env python3
"""Evaluate a trained AudioLLM checkpoint with decoding enabled and save JSON outputs.

Supports resuming a partially-completed evaluation: if the job gets killed by a
cluster wall-clock limit partway through, re-running with --resume will skip
samples that were already evaluated (found in the incremental JSONL file) and
only process the remaining ones, then merge everything into the final result.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import AudioDeepfakeDataset
from train_common import (
    batch_to_samples,
    build_model,
    load_config,
    maybe_hf_login,
    save_config,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint directory")
    parser.add_argument("--epoch", default="best", help="Checkpoint epoch name (e.g. best or 1)")
    parser.add_argument("--split", default="test", choices=["test", "val"], help="Dataset split to evaluate")
    parser.add_argument("--output_dir", default=None, help="Directory to save JSON outputs")
    parser.add_argument("--data_path", default=None, help="Override the dataset JSON used for evaluation")
    parser.add_argument("--hf_token", default=None, help="HF token (or set HF_TOKEN)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples already present in a previous partial run's incremental "
        "JSONL for this split/epoch, and append new results to it.",
    )
    parser.add_argument(
        "--id_key",
        default="path",
        help="Key used to uniquely identify a sample in both the dataset JSON entries "
        "and the saved results (default: 'path').",
    )
    return parser.parse_args()


def build_test_loader(config: dict, split: str, data_path: str | None = None) -> DataLoader:
    data_cfg = config["data"]
    split_path = data_path or data_cfg.get(f"{split}_path")
    if not split_path:
        raise KeyError(f"Missing '{split}_path' in config['data']")

    dataset = AudioDeepfakeDataset(split_path, prompt_path=data_cfg.get("prompt_path"))
    return DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=config["train"].get("num_workers", 4),
        pin_memory=True,
    )


def _load_checkpoint_weights(model, checkpoint_dir: str, epoch: str, device: torch.device) -> None:
    model.to(device)
    model.eval()

    candidates = []
    if epoch != "best":
        candidates.extend([f"lora_{epoch}", f"checkpoint_{epoch}.pt", f"audio_encoder_{epoch}.pt", f"proj_{epoch}.pt"])
    candidates.extend(["lora_best", "checkpoint_best.pt", "audio_encoder_best.pt", "proj_best.pt"])

    if model.speech_proj is not None:
        for name in [f"proj_{epoch}.pt", "proj_best.pt"]:
            path = os.path.join(checkpoint_dir, name)
            if os.path.exists(path):
                logging.info("Loading projector weights from %s", path)
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                model.speech_proj.load_state_dict(state_dict, strict=False)
                break

    if hasattr(model, "audio_encoder") and model.audio_encoder is not None:
        for name in [f"audio_encoder_{epoch}.pt", "audio_encoder_best.pt"]:
            path = os.path.join(checkpoint_dir, name)
            if os.path.exists(path):
                logging.info("Loading audio encoder weights from %s", path)
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                try:
                    model.audio_encoder.load_state_dict(state_dict, strict=False)
                except RuntimeError:
                    fixed_state_dict = {}
                    expected_keys = set(model.audio_encoder.state_dict().keys())
                    for k, v in state_dict.items():
                        if k in expected_keys and model.audio_encoder.state_dict()[k].shape == v.shape:
                            fixed_state_dict[k] = v
                        else:
                            fixed_state_dict[k] = v
                    model.audio_encoder.load_state_dict(fixed_state_dict, strict=False)
                break

    lora_dir = os.path.join(checkpoint_dir, f"lora_{epoch}") if os.path.isdir(os.path.join(checkpoint_dir, f"lora_{epoch}")) else None
    if lora_dir is None:
        lora_dir = os.path.join(checkpoint_dir, "lora_best") if os.path.isdir(os.path.join(checkpoint_dir, "lora_best")) else None

    if lora_dir is not None:
        logging.info("Loading LoRA adapter from %s", lora_dir)
        model.llm_model.load_adapter(lora_dir, adapter_name="default")
    else:
        for name in [f"checkpoint_{epoch}.pt", "checkpoint_best.pt"]:
            path = os.path.join(checkpoint_dir, name)
            if os.path.exists(path):
                logging.info("Loading full checkpoint from %s", path)
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint, strict=False)
                break

    model.to(device)
    model.eval()


def _extract_label(text: str) -> int:
    if text is None:
        return 0
    text_lower = str(text).lower().strip()
    if "<answer>bonafide</answer>" in text_lower or "<answer>real</answer>" in text_lower:
        return 1
    if "<answer>spoof</answer>" in text_lower or "<answer>fake</answer>" in text_lower:
        return 0
    if "<conclusion>bonafide</conclusion>" in text_lower or "<conclusion>real</conclusion>" in text_lower:
        return 1
    if "<conclusion>spoof</conclusion>" in text_lower or "<conclusion>fake</conclusion>" in text_lower:
        return 0
    if "bonafide" in text_lower or "real" in text_lower:
        return 1
    if "spoof" in text_lower or "fake" in text_lower:
        return 0
    return 0


def _compute_classification_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    total_samples = tp + tn + fp + fn
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    balanced_accuracy = 0.5 * (tp / (tp + fn) if (tp + fn) > 0 else 0) + 0.5 * (tn / (tn + fp) if (tn + fp) > 0 else 0)

    precision_positive = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_positive = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_positive = 2 * precision_positive * recall_positive / (precision_positive + recall_positive) if (precision_positive + recall_positive) > 0 else 0

    precision_negative = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_negative = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_negative = 2 * precision_negative * recall_negative / (precision_negative + recall_negative) if (precision_negative + recall_negative) > 0 else 0
    macro_f1 = 0.5 * (f1_positive + f1_negative)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision_positive,
        "recall": recall_positive,
        "f1": f1_positive,
        "precision_negative": precision_negative,
        "recall_negative": recall_negative,
        "f1_negative": f1_negative,
        "macro_f1": macro_f1,
    }


def save_result_incremental(result: list[dict[str, Any]], result_dir: str, filename: str) -> None:
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{filename}.jsonl")
    with open(result_file, "a", encoding="utf-8") as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def merge_incremental_results(result_dir: str, filename: str, final_filename: str) -> list[dict[str, Any]]:
    final_result_file = os.path.join(result_dir, f"{final_filename}.json")
    merged_results = []
    result_file = os.path.join(result_dir, f"{filename}.jsonl")
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    merged_results.append(json.loads(line))
    with open(final_result_file, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)
    return merged_results


def _recompute_counts(results: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for item in results:
        pred_label = _extract_label(item.get("prediction"))
        true_label = _extract_label(item.get("ground_truth"))
        if pred_label == 1 and true_label == 1:
            tp += 1
        elif pred_label == 0 and true_label == 0:
            tn += 1
        elif pred_label == 1 and true_label == 0:
            fp += 1
        elif pred_label == 0 and true_label == 1:
            fn += 1
    return tp, tn, fp, fn


def _load_resume_state(
    output_dir: str, split: str, epoch: str, id_key: str
) -> tuple[set[str], list[dict[str, Any]], bool]:
    """Return (completed_ids, completed_results, is_fully_merged) from a prior run."""
    final_file = os.path.join(output_dir, f"eval_{split}_epoch_{epoch}.json")
    incremental_file = os.path.join(output_dir, f"eval_{split}_epoch_{epoch}_incremental.jsonl")

    if os.path.exists(final_file):
        with open(final_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        completed_ids = {str(item.get("id")) for item in results}
        logging.info("Found completed final results at %s (%d samples).", final_file, len(results))
        return completed_ids, results, True

    if os.path.exists(incremental_file):
        results = []
        with open(incremental_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        completed_ids = {str(item.get("id")) for item in results}
        logging.info(
            "Found partial incremental results at %s (%d samples already done). Resuming.",
            incremental_file,
            len(results),
        )
        return completed_ids, results, False

    return set(), [], False


def _filter_remaining_data(
    data_path: str, completed_ids: set[str], id_key: str, output_dir: str, split: str
) -> str:
    """Write a filtered copy of the dataset JSON excluding already-completed ids."""
    with open(data_path, "r", encoding="utf-8") as f:
        all_items = json.load(f)

    if not isinstance(all_items, list):
        raise ValueError(f"Expected a JSON array in {data_path}, got {type(all_items)}")

    remaining = [item for item in all_items if str(item.get(id_key)) not in completed_ids]

    logging.info(
        "Resume filter: %d total, %d already done, %d remaining.",
        len(all_items),
        len(all_items) - len(remaining),
        len(remaining),
    )

    filtered_path = os.path.join(output_dir, f"_resume_{split}_remaining.json")
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(remaining, f, ensure_ascii=False)

    return filtered_path


def evaluate_split(
    model,
    loader: DataLoader,
    device: torch.device,
    generate_cfg: dict[str, Any],
    output_dir: str,
    split: str,
    epoch: str,
    save_json: bool,
    save_freq: int,
    prior_results: list[dict[str, Any]],
) -> dict[str, Any]:
    model.eval()
    results: list[dict[str, Any]] = []
    tp, tn, fp, fn = _recompute_counts(prior_results)
    total_samples = len(prior_results)
    total_loss = 0.0
    batch_count = 0

    for batch_idx, batch in enumerate(loader):
        samples = batch_to_samples(batch, device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            with torch.no_grad():
                forward_result = model(samples, verbose=True)

        loss = forward_result.get("loss", 0.0)
        total_loss += loss.item()

        prompts = batch["prompt"]
        generated_texts = model.generate(samples, generate_cfg, prompts=prompts)

        ground_truth_texts = batch["text"]
        for i, (pred_text, gt_text) in enumerate(zip(generated_texts, ground_truth_texts)):
            pred_label = _extract_label(pred_text)
            true_label = _extract_label(gt_text)
            if pred_label == 1 and true_label == 1:
                tp += 1
            elif pred_label == 0 and true_label == 0:
                tn += 1
            elif pred_label == 1 and true_label == 0:
                fp += 1
            elif pred_label == 0 and true_label == 1:
                fn += 1
            total_samples += 1

        for item_id, pred_text, gt_text, prompt in zip(batch.get("path", [""] * len(ground_truth_texts)), generated_texts, ground_truth_texts, prompts):
            results.append(
                {
                    "id": item_id,
                    "ground_truth": gt_text,
                    "prediction": pred_text,
                    "prompt": prompt,
                }
            )

        batch_count += 1
        if save_json and batch_count % save_freq == 0:
            save_result_incremental(results, output_dir, f"eval_{split}_epoch_{epoch}_incremental")
            logging.info("Checkpoint: saved %d new results (running total incl. prior: %d)", len(results), total_samples)
            results = []

    if save_json:
        if results:
            save_result_incremental(results, output_dir, f"eval_{split}_epoch_{epoch}_incremental")
        merge_incremental_results(output_dir, f"eval_{split}_epoch_{epoch}_incremental", f"eval_{split}_epoch_{epoch}")

    if total_samples > 0:
        metrics = _compute_classification_metrics(tp, tn, fp, fn)
        result = {
            "loss": total_loss / max(1, batch_count) if batch_count > 0 else None,
            **metrics,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "total_samples": total_samples,
        }
        if batch_count == 0:
            result["note"] = "No new batches were run this invocation; metrics are from prior run(s) only."
        elif prior_results:
            result["note"] = (
                f"loss is averaged over the {batch_count} batch(es) run in this invocation only "
                f"(per-sample loss from {len(prior_results)} prior-run samples was not persisted); "
                "classification metrics (accuracy/f1/etc.) cover all samples, old and new."
            )
        return result

    return {"loss": total_loss / max(1, len(loader)), "accuracy": 0.0}


def main() -> None:
    args = parse_args()
    maybe_hf_login(args.hf_token)

    set_seed(args.seed)
    config = load_config(args.config)
    output_dir = args.output_dir or os.path.join(Path(args.checkpoint_dir).parent, f"{Path(args.checkpoint_dir).name}_eval")
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir, log_name="test.log")
    save_config(config, output_dir)

    completed_ids: set[str] = set()
    prior_results: list[dict[str, Any]] = []
    effective_data_path = args.data_path

    if args.resume:
        completed_ids, prior_results, is_complete = _load_resume_state(
            output_dir, args.split, args.epoch, args.id_key
        )
        if is_complete:
            metrics = _compute_classification_metrics(*_recompute_counts(prior_results))
            metrics["total_samples"] = len(prior_results)
            logging.info(
                "Evaluation for split=%s epoch=%s is already fully complete (%d samples). "
                "Skipping model load and generation. Metrics: %s",
                args.split,
                args.epoch,
                len(prior_results),
                json.dumps(metrics, indent=2),
            )
            return

        if completed_ids:
            data_cfg = config["data"]
            base_data_path = args.data_path or data_cfg.get(f"{args.split}_path")
            if not base_data_path:
                raise KeyError(f"Missing '{args.split}_path' in config['data']")
            effective_data_path = _filter_remaining_data(
                base_data_path, completed_ids, args.id_key, output_dir, args.split
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    model = build_model(config["model"])
    _load_checkpoint_weights(model, args.checkpoint_dir, args.epoch, device)

    loader = build_test_loader(config, args.split, data_path=effective_data_path)
    generate_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    metrics = evaluate_split(
        model,
        loader,
        device,
        generate_cfg,
        output_dir,
        args.split,
        args.epoch,
        save_json=True,
        save_freq=args.save_freq,
        prior_results=prior_results,
    )
    logging.info("Evaluation metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()