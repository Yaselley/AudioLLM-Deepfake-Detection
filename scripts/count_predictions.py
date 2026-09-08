#!/usr/bin/env python3
"""
Count how many times each exact 'prediction' string occurs, across all
.json files in a directory. Analyzes each file individually, then also
produces an aggregated count across all files combined.

Each JSON file is expected to be a list of records like:
[
  {"id": "...", "ground_truth": "...", "prediction": "...", "prompt": "..."},
  ...
]
(or a dict wrapping such a list under a key like "data"/"records"/"items"/"results")

Usage:
    python count_predictions_dir.py /path/to/dir [--top N] [--out-dir results] [--field prediction] [--pattern "*.json"]

Outputs:
    - Printed per-file summary for each .json file found
    - Printed aggregated summary across all files
    - If --out-dir is given: one CSV per file + one "aggregated.csv"
"""

import json
import argparse
import csv
import glob
import os
from collections import Counter


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "records", "items", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError(f"{path}: JSON is an object but no list of records found inside it")
    if not isinstance(data, list):
        raise ValueError(f"{path}: Expected a JSON list of records")
    return data


def extract_values(records, field):
    values, missing = [], 0
    for r in records:
        v = r.get(field) if isinstance(r, dict) else None
        if v is None:
            missing += 1
        else:
            values.append(v)
    return values, missing


def print_summary(label, values, missing, total, top_n):
    counts = Counter(values)
    unique = len(counts)

    print(f"\n=== {label} ===")
    print(f"Total records:          {total}")
    print(f"Records with field:     {len(values)}")
    if missing:
        print(f"Records missing field:  {missing}")
    print(f"Unique strings:         {unique}")
    if values:
        print(f"Duplicate rate:         {(1 - unique / len(values)) * 100:.1f}%")

    most_common = counts.most_common(top_n)
    if most_common:
        print(f"\n{'Count':>6}  {'%':>6}  Preview")
        print("-" * 90)
        for text, cnt in most_common:
            pct = cnt / len(values) * 100
            preview = text.replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:77] + "..."
            print(f"{cnt:>6}  {pct:5.1f}%  {preview}")

    return counts


def write_csv(path, counts, values, field):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["count", "percentage", field])
        for text, cnt in counts.most_common():
            pct = cnt / len(values) * 100 if values else 0
            writer.writerow([cnt, f"{pct:.2f}", text])


def main():
    parser = argparse.ArgumentParser(description="Count exact-duplicate prediction strings across a directory of JSON files")
    parser.add_argument("directory", help="Directory containing JSON files")
    parser.add_argument("--pattern", default="*.json", help="Glob pattern for files to include (default: *.json)")
    parser.add_argument("--top", type=int, default=None, help="Only show top N most common predictions per report")
    parser.add_argument("--out-dir", default=None, help="Optional directory to write per-file and aggregated CSVs")
    parser.add_argument("--field", default="prediction", help="Field name to count (default: prediction)")
    parser.add_argument("--no-recursive", action="store_true", help="Only look in the top-level directory, not subdirectories")
    args = parser.parse_args()

    if args.no_recursive:
        paths = sorted(glob.glob(os.path.join(args.directory, args.pattern)))
    else:
        paths = sorted(glob.glob(os.path.join(args.directory, "**", args.pattern), recursive=True))
    if not paths:
        print(f"No files matching {args.pattern} found in {args.directory}")
        return

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    all_values = []
    all_missing = 0
    all_total = 0

    for path in paths:
        rel = os.path.relpath(path, args.directory)
        try:
            records = load_records(path)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"\n=== {rel} ===")
            print(f"SKIPPED - could not parse: {e}")
            continue

        values, missing = extract_values(records, args.field)
        total = len(records)

        counts = print_summary(rel, values, missing, total, args.top)

        if args.out_dir:
            safe_name = rel.replace(os.sep, "__")
            out_path = os.path.join(args.out_dir, os.path.splitext(safe_name)[0] + "_counts.csv")
            write_csv(out_path, counts, values, args.field)

        all_values.extend(values)
        all_missing += missing
        all_total += total

    agg_counts = print_summary(f"AGGREGATED ({len(paths)} files)", all_values, all_missing, all_total, args.top)

    if args.out_dir:
        agg_path = os.path.join(args.out_dir, "aggregated_counts.csv")
        write_csv(agg_path, agg_counts, all_values, args.field)
        print(f"\nCSV files written to: {args.out_dir}/")


if __name__ == "__main__":
    main()