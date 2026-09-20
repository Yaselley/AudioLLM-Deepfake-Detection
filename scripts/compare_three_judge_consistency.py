#!/usr/bin/env python3
"""Compare the three judge models used for HIR-SDD consistency analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_FILES = (
    ("gpt-oss-120b", "judge_scores_openai_gpt-oss-120b.jsonl"),
    ("gemma-4-31B", "judge_scores_google_gemma-4-31B.jsonl"),
    ("deepseek-v4-flash", "judge_scores_deepseek-ai_DeepSeek-V4-Flash.jsonl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True, help="Directory containing judge score JSONL files")
    parser.add_argument("--output", type=Path, help="Optional JSON path for aggregated consistency summary")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def score_for_record(record: dict[str, Any]) -> float | None:
    judge_summary = record.get("judge_summary")
    if isinstance(judge_summary, dict):
        value = judge_summary.get("mean_score")
        if value is not None:
            return float(value)

    judge = record.get("judge")
    if isinstance(judge, dict):
        scores = []
        for criterion_data in judge.values():
            if isinstance(criterion_data, dict):
                score = criterion_data.get("score")
                if score is not None:
                    scores.append(float(score))
        if scores:
            return sum(scores) / len(scores)
    return None


def model_scores(eval_dir: Path) -> dict[str, list[float | None]]:
    result: dict[str, list[float | None]] = {}
    for model_name, filename in MODEL_FILES:
        path = eval_dir / filename
        records = read_jsonl(path)
        values = [score_for_record(record) for record in records]
        result[model_name] = values
    return result


def pairwise_summary(model_values: dict[str, list[float | None]]) -> list[dict[str, Any]]:
    pairwise: list[dict[str, Any]] = []
    model_names = list(model_values.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            left_name, right_name = model_names[i], model_names[j]
            left_values = model_values[left_name]
            right_values = model_values[right_name]
            diffs: list[float] = []
            for left, right in zip(left_values, right_values):
                if left is None or right is None:
                    continue
                diffs.append(abs(float(left) - float(right)))
            pairwise.append({
                "models": [left_name, right_name],
                "avg_abs_diff": sum(diffs) / len(diffs) if diffs else None,
                "max_abs_diff": max(diffs) if diffs else None,
                "n_records": len(diffs),
            })
    return pairwise


def overall_means(model_values: dict[str, list[float | None]]) -> dict[str, float | None]:
    means: dict[str, float | None] = {}
    for model_name, values in model_values.items():
        finite = [float(v) for v in values if v is not None]
        means[model_name] = sum(finite) / len(finite) if finite else None
    return means


def main() -> None:
    args = parse_args()
    scores = model_scores(args.eval_dir)
    summary = {
        "models": list(scores.keys()),
        "means": overall_means(scores),
        "pairwise_abs_diff": pairwise_summary(scores),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
