#!/usr/bin/env python3
"""Evaluate predictions that have already been saved to JSON or JSONL.

Each record should contain ``ground_truth`` and ``prediction``. Predictions
may be plain labels or generated text containing an ``<answer>`` or
``<conclusion>`` tag.

Examples:
    python evaluate_jsonl.py results.jsonl
    python evaluate_jsonl.py results.jsonl --output metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


REAL_LABELS = ("bonafide", "real", "authentic", "genuine")
FAKE_LABELS = ("spoof", "fake", "synthetic", "artificial", "generated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Computed prediction JSONL or JSON file")
    parser.add_argument("--output", type=Path, help="Optional path for the metrics JSON")
    parser.add_argument("--ground-truth-key", default="ground_truth")
    parser.add_argument("--prediction-key", default="prediction")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON, a JSON array, or a wrapped record list."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl" or not text.startswith(("[", "{")):
        data: Any = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)

    if isinstance(data, dict):
        for key in ("data", "records", "items", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            raise ValueError(f"{path}: expected a list of records")

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{path}: expected a list of JSON objects")
    return data


def extract_label(value: Any) -> int | None:
    """Return 1 for real, 0 for fake, or None when no label is recognizable."""
    if value is None:
        return None
    text = str(value)
    for tag in ("answer", "conclusion"):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1)
            break
    text = text.strip().lower()

    if any(label in text for label in REAL_LABELS):
        return 1
    if any(label in text for label in FAKE_LABELS) or "not genuine" in text:
        return 0
    return None


def evaluate(records: Iterable[dict[str, Any]], ground_truth_key: str, prediction_key: str) -> dict[str, Any]:
    tp = tn = fp = fn = skipped = 0
    for item in records:
        true_label = extract_label(item.get(ground_truth_key))
        pred_label = extract_label(item.get(prediction_key))
        if true_label is None or pred_label is None:
            skipped += 1
        elif pred_label == 1 and true_label == 1:
            tp += 1
        elif pred_label == 0 and true_label == 0:
            tn += 1
        elif pred_label == 1:
            fp += 1
        else:
            fn += 1

    evaluated = tp + tn + fp + fn
    positive_precision = tp / (tp + fp) if tp + fp else 0.0
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    positive_f1 = (
        2 * positive_precision * positive_recall / (positive_precision + positive_recall)
        if positive_precision + positive_recall
        else 0.0
    )
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )

    return {
        "total_records": evaluated + skipped,
        "evaluated_records": evaluated,
        "skipped_records": skipped,
        "accuracy": (tp + tn) / evaluated if evaluated else 0.0,
        "balanced_accuracy": 0.5 * (
            (tp / (tp + fn) if tp + fn else 0.0) + (tn / (tn + fp) if tn + fp else 0.0)
        ),
        "precision": positive_precision,
        "recall": positive_recall,
        "f1": positive_f1,
        "precision_negative": negative_precision,
        "recall_negative": negative_recall,
        "f1_negative": negative_f1,
        "macro_f1": 0.5 * (positive_f1 + negative_f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    metrics = evaluate(load_records(args.input), args.ground_truth_key, args.prediction_key)
    output = json.dumps(metrics, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()