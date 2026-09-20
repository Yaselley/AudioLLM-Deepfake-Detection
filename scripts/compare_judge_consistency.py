#!/usr/bin/env python3
"""Compare multiple judge score files for cross-model consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-files", nargs="+", required=True, help="Judge score JSONL files to compare")
    parser.add_argument("--output", type=Path, help="Optional output path for the JSON summary")
    return parser.parse_args()


def read_scores(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_model_scores(score_files: list[Path]) -> dict[str, list[float]]:
    model_scores: dict[str, list[float]] = {}
    for score_file in score_files:
        model_name = score_file.stem.replace("judge_scores_", "")
        records = read_scores(score_file)
        scores: list[float] = []
        for record in records:
            judge_summary = record.get("judge_summary", {})
            mean_score = judge_summary.get("mean_score")
            if mean_score is not None:
                scores.append(float(mean_score))
        model_scores[model_name] = scores
    return model_scores


def summarize(model_scores: dict[str, list[float]]) -> dict[str, Any]:
    if not model_scores:
        return {"models": [], "pairwise_abs_diff": [], "global_mean": None}

    length = max(len(scores) for scores in model_scores.values()) if model_scores else 0
    aligned: dict[str, list[float]] = {}
    for model, scores in model_scores.items():
        aligned[model] = scores[:length]
        if len(aligned[model]) < length:
            aligned[model].extend([None] * (length - len(aligned[model])))

    pairwise: list[dict[str, Any]] = []
    model_names = list(model_scores.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a_name, b_name = model_names[i], model_names[j]
            a_scores = aligned[a_name]
            b_scores = aligned[b_name]
            diffs: list[float] = []
            for left, right in zip(a_scores, b_scores):
                if left is None or right is None:
                    continue
                diffs.append(abs(float(left) - float(right)))
            pairwise.append({
                "models": [a_name, b_name],
                "avg_abs_diff": sum(diffs) / len(diffs) if diffs else None,
                "max_abs_diff": max(diffs) if diffs else None,
                "n_records": len(diffs),
            })

    global_means: dict[str, float | None] = {}
    for model, scores in model_scores.items():
        finite = [float(v) for v in scores if v is not None]
        global_means[model] = sum(finite) / len(finite) if finite else None

    return {
        "models": model_names,
        "model_mean_scores": global_means,
        "pairwise_abs_diff": pairwise,
        "global_mean": sum(v for v in global_means.values() if v is not None) / len([v for v in global_means.values() if v is not None]) if any(v is not None for v in global_means.values()) else None,
    }


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.score_files]
    model_scores = build_model_scores(paths)
    summary = summarize(model_scores)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
