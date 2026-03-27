#!/usr/bin/env python3
"""
Build JSONL for Qwen-Omni thinking mode from data_thinking (swift/audio-only).

Creates training data similar to data_swift/asv19_train.jsonl but with assistant
responses in Qwen-Omni thinking format. No OpenSMILE features - just audio + prompt.

User: "<audio>Is this audio real or fake?"
Assistant: <think>{reasoning}</think>\n<answer>{prediction}</answer>

Input: data_thinking/LA_T_*/result.json (with thinking, prediction)
Output: data_swift/asv19_train_thinking.jsonl
"""

import json
from pathlib import Path


def build_assistant_thinking(thinking: str, prediction: str) -> str:
    """
    Build assistant content in Qwen-Omni thinking format.
    """
    prediction_clean = prediction.strip()
    return f"<think>\n{thinking}\n</think>\n<answer>{prediction_clean}</answer>"


def main():
    base = Path(__file__).resolve().parent.parent
    thinking_dir = base / "data" / "data_thinking"
    output_path = base / "data" / "data_swift" / "asv19_train_thinking.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(d for d in thinking_dir.iterdir() if d.is_dir())
    written = 0
    skipped = 0

    with open(output_path, "w") as f:
        for sample_dir in sample_dirs:
            result_file = sample_dir / "result.json"
            if not result_file.exists():
                skipped += 1
                continue

            with open(result_file) as rf:
                data = json.load(rf)

            thinking = data.get("thinking")
            prediction = data.get("prediction")
            audio_path = data.get("audio", "")

            if not thinking or not prediction:
                skipped += 1
                continue

            user_content = "<audio>Is this audio real or fake?"
            assistant_content = build_assistant_thinking(thinking, prediction)

            record = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "audios": [audio_path],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} samples to {output_path}")
    if skipped:
        print(f"Skipped {skipped} samples (missing result.json or required fields)")


if __name__ == "__main__":
    main()
