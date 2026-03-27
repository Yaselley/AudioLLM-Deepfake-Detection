#!/usr/bin/env python3
"""
Build thinking-mode JSONL train/val data from data_thinking/ and existing asv19 splits.

Output format (per line):
  {"path": "...", "label": "real", "thinking": "<think>...\n</think>\n<answer>Real</answer>"}

- For samples WITH thinking data: full <think>...\n</think>\n<answer>...</answer>
- For samples WITHOUT thinking data: thinking field omitted (dataset falls back to simple label)

Produces:
  data/asv19_train_thinking.jsonl  (from data_thinking LA_T_* matched to asv19_train)
  data/asv19_val_thinking.jsonl    (validation — no thinking available, plain labels only)
"""

import json
import os
from pathlib import Path


def load_thinking_map(thinking_dir):
    """Build audio_id → {thinking, prediction, audio_path} from data_thinking."""
    thinking_map = {}
    for sample_dir in sorted(thinking_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        result_file = sample_dir / "result.json"
        if not result_file.exists():
            continue
        with open(result_file) as f:
            data = json.load(f)
        thinking = data.get("thinking")
        prediction = data.get("prediction")
        audio_path = data.get("audio", "")
        if not thinking or not prediction:
            continue

        audio_id = sample_dir.name
        thinking_map[audio_id] = {
            "thinking": thinking,
            "prediction": prediction,
            "audio_path": audio_path,
        }
    return thinking_map


def build_thinking_target(thinking_text, prediction):
    """Build the full target string: <think>...\n</think>\n<answer>...</answer>"""
    label = prediction.strip()
    return f"<think>\n{thinking_text}\n</think>\n<answer>{label}</answer>"


def process_split(input_jsonl, thinking_map, output_jsonl, split_prefix):
    """
    Process a split JSONL.
    Match each sample by audio_id. If thinking exists, include it.
    The target text (samples["text"]) is the full thinking string.
    label is kept as metadata only.
    """
    with_thinking = 0
    without_thinking = 0

    with open(input_jsonl) as f_in, open(output_jsonl, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio_path = item["path"]
            label = item["label"]

            filename = os.path.splitext(os.path.basename(audio_path))[0]
            audio_id = filename

            record = {
                "path": audio_path,
                "label": label,
            }

            if audio_id in thinking_map:
                t = thinking_map[audio_id]
                record["thinking"] = build_thinking_target(t["thinking"], t["prediction"])
                with_thinking += 1
            else:
                # No thinking data — target will be simple "Real"/"Fake" (from label)
                without_thinking += 1

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  {split_prefix}: {with_thinking} with thinking, {without_thinking} without → {output_jsonl}")


def main():
    base = Path(__file__).resolve().parent.parent
    thinking_dir = base / "data" / "data_thinking"
    data_dir = base / "data"

    print("Loading thinking data...")
    thinking_map = load_thinking_map(thinking_dir)
    print(f"  Found {len(thinking_map)} samples with thinking\n")

    # Train split
    process_split(
        data_dir / "asv19_train.jsonl",
        thinking_map,
        data_dir / "asv19_train_thinking.jsonl",
        "train"
    )

    # Validation split (LA_D_* — no thinking available, but we still create the file)
    process_split(
        data_dir / "asv19_val.jsonl",
        thinking_map,
        data_dir / "asv19_val_thinking.jsonl",
        "val"
    )


if __name__ == "__main__":
    main()
