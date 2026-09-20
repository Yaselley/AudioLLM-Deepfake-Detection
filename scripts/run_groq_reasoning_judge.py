#!/usr/bin/env python3
"""Call Groq on prepared HIR-SDD judge prompts and save JSONL responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from groq import Groq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", type=Path, help="Prepared judge_prompts.json")
    parser.add_argument("--output", type=Path, required=True, help="Raw Groq responses JSONL")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--resume", action="store_true", help="Keep existing complete output records")
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{path}: expected a JSON array of prompt objects")
    return data


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    existing = load_existing(args.output) if args.resume else []
    if len(existing) > len(prompts):
        raise ValueError(f"{args.output}: has more records than {args.prompts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if existing else "w"
    client = Groq()
    with args.output.open(mode, encoding="utf-8") as handle:
        for index in range(len(existing), len(prompts)):
            prompt = prompts[index]
            completion = client.chat.completions.create(
                model=args.model,
                messages=prompt["messages"],
                temperature=args.temperature,
                max_completion_tokens=args.max_completion_tokens,
                top_p=args.top_p,
                reasoning_effort=args.reasoning_effort,
                stream=True,
                stop=None,
            )
            content_parts: list[str] = []
            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta:
                    content_parts.append(delta)
            result = {
                "record_index": prompt.get("record_index"),
                "criterion": prompt.get("criterion"),
                "response": "".join(content_parts),
                "model": args.model,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"completed {index + 1}/{len(prompts)}", flush=True)


if __name__ == "__main__":
    main()
