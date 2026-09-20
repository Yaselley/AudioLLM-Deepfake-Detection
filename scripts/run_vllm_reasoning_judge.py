#!/usr/bin/env python3
"""Run the prepared HIR-SDD judge prompts with a local vLLM model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()

def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{path}: expected a JSON array")
    return data


def main() -> None:
    args = parse_args()
    prompts = load_records(args.prompts)
    existing: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        existing = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(existing) > len(prompts):
        raise ValueError(f"{args.output}: has more records than {args.prompts}")
    pending = prompts[len(existing):]
    if not pending:
        print(f"Already complete: {args.output}")
        return

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outputs = llm.chat([prompt["messages"] for prompt in pending], sampling_params=sampling_params)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if existing else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for prompt, output in zip(pending, outputs):
            text = output.outputs[0].text if output.outputs else ""
            handle.write(json.dumps({
                "record_index": prompt.get("record_index"),
                "response": text,
                "model": args.model,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pending)} responses to {args.output}")


if __name__ == "__main__":
    main()
