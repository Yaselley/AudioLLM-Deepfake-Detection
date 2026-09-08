#!/usr/bin/env python3
"""
Replace the <conclusion>...</conclusion> and <answer>...</answer> values in
a rewritten-reasoning JSON file with the ground-truth label pulled from the
matching record in the ORIGINAL reasoning JSON file (same "path"/"text"/
"task" structure).

This does NOT touch <observations> or <reasoning_step> content -- only the
conclusion and answer tags are swapped, in case the judge/rewrite pipeline
introduced a wrong label.

Usage:
    python fix_labels.py \
        --new-file hir_sdd_train_reasoning.json \
        --original-file original_reasoning.json \
        --output-file hir_sdd_train_reasoning_fixed.json
"""

import argparse
import json
import re
import sys

CONCLUSION_RE = re.compile(r"(<conclusion>)(.*?)(</conclusion>)", re.DOTALL)
ANSWER_RE = re.compile(r"(<answer>)(.*?)(</answer>)", re.DOTALL)


def load_json_array(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array at the top level")
    return data


def extract_label(text):
    """Pull the ground-truth label out of a record's text -- prefer
    <answer>, fall back to <conclusion> if answer is missing."""
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if m:
        return m.group(2).strip()
    m = CONCLUSION_RE.search(text)
    if m:
        return m.group(2).strip()
    return None


def replace_label(text, new_label):
    """Overwrite conclusion and answer tag contents with new_label,
    leaving everything else in the text unchanged. If <answer> is missing
    entirely, append a new <answer>new_label</answer> at the end. If
    <conclusion> is missing, it is left as-is (not added)."""
    if not text or new_label is None:
        return text, False

    text = CONCLUSION_RE.sub(lambda m: m.group(1) + new_label + m.group(3), text, count=1)

    if ANSWER_RE.search(text):
        text = ANSWER_RE.sub(lambda m: m.group(1) + new_label + m.group(3), text, count=1)
        answer_added = False
    else:
        text = text.rstrip() + f"\n<answer>{new_label}</answer>"
        answer_added = True

    return text, answer_added


def build_path_index(records, path_field, text_field):
    index = {}
    for rec in records:
        p = rec.get(path_field)
        if p is None:
            continue
        index.setdefault(p, []).append(rec)
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-file", required=True, help="Rewritten-reasoning JSON file whose labels need fixing")
    ap.add_argument("--original-file", required=True, help="Original reasoning JSON file holding ground-truth labels")
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--path-field", default="path")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--by-order", action="store_true", help="Match by position instead of by 'path' (use if paths aren't unique/reliable)")
    args = ap.parse_args()

    new_records = load_json_array(args.new_file)
    original_records = load_json_array(args.original_file)

    print(f"Loaded {len(new_records)} new records and {len(original_records)} original records.", file=sys.stderr)

    path_index = None if args.by_order else build_path_index(original_records, args.path_field, args.text_field)

    fixed = 0
    no_match = 0
    no_label_in_original = 0
    unchanged = 0
    answer_added = 0

    output_records = []
    for i, rec in enumerate(new_records):
        orig_rec = None

        if not args.by_order:
            p = rec.get(args.path_field)
            candidates = path_index.get(p) if p is not None else None
            if candidates:
                orig_rec = candidates.pop(0)

        if orig_rec is None:
            if i < len(original_records):
                orig_rec = original_records[i]
            else:
                no_match += 1
                print(f"Warning: no matching original record for #{i} (path={rec.get(args.path_field)}), leaving unchanged.", file=sys.stderr)
                output_records.append(rec)
                continue

        orig_label = extract_label(orig_rec.get(args.text_field))
        if orig_label is None:
            no_label_in_original += 1
            print(f"Warning: could not extract label from original record #{i} (path={rec.get(args.path_field)}), leaving unchanged.", file=sys.stderr)
            output_records.append(rec)
            continue

        new_text = rec.get(args.text_field)
        replaced, was_answer_added = replace_label(new_text, orig_label)
        if replaced == new_text:
            unchanged += 1
        if was_answer_added:
            answer_added += 1

        new_rec = dict(rec)
        new_rec[args.text_field] = replaced
        output_records.append(new_rec)
        fixed += 1

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"Wrote {len(output_records)} records to {args.output_file} "
        f"({fixed} labels fixed, {answer_added} had a missing <answer> tag added, "
        f"{unchanged} left unchanged (already correct), {no_match} unmatched, "
        f"{no_label_in_original} originals had no extractable label).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()