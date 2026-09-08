#!/usr/bin/env python3
"""
Move the <answer>...</answer> tag to the very beginning of each record's
"text" field, ahead of the <reasoning> block, leaving everything else
(observations, reasoning_step, conclusion) untouched in its original order.

    Before: <reasoning>...</reasoning>\n<answer>Fake</answer>
    After:  <answer>Fake</answer>\n<reasoning>...</reasoning>

Usage:
    python move_answer_to_front.py \
        --input-file hir_sdd_train_reasoning_fixed.json \
        --output-file hir_sdd_train_reasoning_front.json
"""

import argparse
import json
import re
import sys

ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def move_answer_to_front(text, separator="\n"):
    """Extract the <answer>...</answer> block and move it to the front of
    text. Returns text unchanged if no <answer> tag is found, or if the
    <answer> tag is already at the very front."""
    if not text:
        return text

    m = ANSWER_RE.search(text)
    if not m:
        return text

    answer_block = m.group(0)
    remainder = (text[: m.start()] + text[m.end():]).strip()

    if not remainder:
        return answer_block

    return answer_block + separator + remainder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-file", required=True, help="JSON array with a text field to fix")
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--separator", default="\n", help="String inserted between the moved <answer> tag and the rest of the text (default: newline)")
    args = ap.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        print(f"Error: {args.input_file} is not a JSON array at the top level.", file=sys.stderr)
        sys.exit(1)

    moved = 0
    no_answer_tag = 0
    already_front = 0

    for rec in records:
        text = rec.get(args.text_field)
        if not text:
            continue

        m = ANSWER_RE.search(text)
        if not m:
            no_answer_tag += 1
            continue

        if m.start() == 0:
            already_front += 1
            continue

        rec[args.text_field] = move_answer_to_front(text, args.separator)
        moved += 1

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"Wrote {len(records)} records to {args.output_file} "
        f"({moved} moved, {already_front} already at front, {no_answer_tag} had no <answer> tag).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()