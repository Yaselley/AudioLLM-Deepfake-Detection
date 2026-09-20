#!/usr/bin/env python3
"""Prepare and merge HIR-SDD reasoning scores from a Groq LLM judge."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

CRITERIA = {
    "correctness": "Rate how well the provided Answer matches the Reference conclusion.",
    "reasoning_quality": "Rate the reasoning quality. Good answers provide detailed analysis matching the reference.",
    "comprehensiveness": "Did the Answer cover all the key points mentioned in the Reference?",
    "logic": "Assess the logic of the answer.",
}
DEFAULT_PREDICTION_KEYS = ("prediction", "model_answer", "rewritten_output", "output", "generation", "response", "completion")
DEFAULT_REFERENCE_KEYS = ("ground_truth", "reference", "gt_text", "ground_truth_text", "original_text", "text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="HIR-SDD test records in JSON or JSONL format")
    parser.add_argument("--output", type=Path, required=True, help="Merged per-record scores JSONL")
    parser.add_argument("--prediction-key", help="Field containing the generated answer")
    parser.add_argument("--reference-key", help="Field containing the reference answer/reasoning")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--prompt-file", type=Path, help="JSON file for the prepared judge prompts")
    parser.add_argument("--judge-output", type=Path, help="Raw local judge output; required during merge")
    parser.add_argument("--prepare-only", action="store_true", help="Write prompts only; do not call the model")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N records")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data: Any = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix.lower() == ".jsonl" else json.loads(text)
    if isinstance(data, dict):
        for key in ("data", "records", "items", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{path}: expected a JSON array or JSONL objects")
    return data


def find_field(record: dict[str, Any], requested: str | None, defaults: tuple[str, ...], label: str) -> tuple[str, str]:
    keys = (requested,) if requested else defaults
    for key in keys:
        if key and key in record and record[key] is not None:
            return str(record[key]), key
    raise KeyError(f"no {label} field found; tried {', '.join(key for key in keys if key)}")


def build_prompt(reference: str, prediction: str) -> str:
    return (
        f"{SYSTEM_PROMPT} USER: You are a strict judge. Evaluate the Answer on all four criteria:\n"
        "1. Correctness: how well the Answer matches the Reference conclusion.\n"
        "2. Reasoning quality: detailed analysis matching the Reference.\n"
        "3. Comprehensiveness: coverage of all key points in the Reference.\n"
        "4. Logic: coherence between the evidence and the conclusion.\n\n"
        f"Reference: {reference}\n\nAnswer: {prediction}\n\n"
        "Output ONLY these four lines, with integer scores from 1 to 10:\n"
        "Correctness: X\nReasoning quality: Y\nComprehensiveness: Z\nLogic: W\n"
        "ASSISTANT:"
    )


def write_swift_prompts(records: list[dict[str, Any]], path: Path, prediction_key: str | None, reference_key: str | None) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        prediction, _ = find_field(record, prediction_key, DEFAULT_PREDICTION_KEYS, "prediction")
        reference, _ = find_field(record, reference_key, DEFAULT_REFERENCE_KEYS, "reference")
        prompts.append({
            "record_index": record_index,
            "messages": [{"role": "user", "content": build_prompt(reference, prediction)}],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return prompts


def parse_scores(text: str) -> dict[str, float | None]:
    labels = {
        "correctness": r"correctness",
        "reasoning_quality": r"reasoning\s+quality",
        "comprehensiveness": r"comprehensiveness",
        "logic": r"logic",
    }
    scores = {criterion: None for criterion in CRITERIA}
    for criterion, label in labels.items():
        match = re.search(rf"{label}\s*[:=]\s*(10|[1-9])(?!\d)", text or "", re.IGNORECASE)
        if match:
            scores[criterion] = float(match.group(1))
    return scores


def response_text(record: dict[str, Any]) -> str:
    for key in ("response", "generated_text", "output", "text", "prediction", "choices"):
        value = record.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return str(value[0].get("message", {}).get("content", value[0]))
    raise KeyError("could not find generated text in MS-Swift output record")


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError(f"{args.input}: no records found")

    if args.prepare_only and not args.prompt_file:
        raise ValueError("--prompt-file is required with --prepare-only")
    if not args.prepare_only and not args.judge_output:
        raise ValueError("model inference is intentionally separate; provide --judge-output to merge results")
    prompt_file = args.prompt_file or args.output.with_name(f"{args.output.stem}_prompts.json")
    prompts = write_swift_prompts(records, prompt_file, args.prediction_key, args.reference_key)

    if args.prepare_only:
        print(json.dumps({"records": len(records), "judge_calls": len(prompts), "prompt_file": str(prompt_file)}, indent=2))
        return

    judge_records = load_records(args.judge_output)
    if len(judge_records) != len(prompts):
        raise ValueError(f"Judge returned {len(judge_records)} records for {len(prompts)} prompts")

    grouped: list[dict[str, Any]] = [dict(record) for record in records]
    for prompt, judge_record in zip(prompts, judge_records):
        judge_response = response_text(judge_record)
        scores = parse_scores(judge_response)
        judge = grouped[prompt["record_index"]].setdefault("judge", {})
        for criterion, score in scores.items():
            judge[criterion] = {
            "response": judge_response,
                "score": score,
            }
    for record in grouped:
        scores = [item["score"] for item in record["judge"].values() if item["score"] is not None]
        record["judge_summary"] = {"model": args.model, "mean_score": sum(scores) / len(scores) if scores else None}

    aspect_means = {}
    for criterion in CRITERIA:
        criterion_scores = [
            record["judge"][criterion]["score"]
            for record in grouped
            if record["judge"][criterion]["score"] is not None
        ]
        aspect_means[criterion] = sum(criterion_scores) / len(criterion_scores) if criterion_scores else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in grouped:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {"records": len(grouped), "judge_calls": len(prompts), "aspect_means": aspect_means, "output": str(args.output)}
    summary_text = json.dumps(summary, indent=2)

    # print in parent/test.log
    log_path = args.output.parent / "test.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(summary_text + "\n")

    print(summary_text)


if __name__ == "__main__":
    main()