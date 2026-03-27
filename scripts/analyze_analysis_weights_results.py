#!/usr/bin/env python3
"""
Script to analyze all results from outputs/analysis_weights (or opensmile/after, etc.).
Uses config.json in each run directory to get seed, LLM, audio encoder, etc.

Metric file naming: use --metrics_mode to choose:
  - new:  Only NEW* files (NEW{dataset}_best_42NEW_metrics_best_42.json, NEW{dataset}_best_42.csv).
  - old:  Only non-NEW files ({dataset}_best_42_metrics_best_42.json, {dataset}_best_42.csv).
Use "old" for folders that have only the classic best_42 naming (no NEW prefix).

Outputs: summary CSV, averaged per seed, across seeds, per-dataset avg/std; optional JSON.
"""

import argparse
import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import det_curve, f1_score, accuracy_score


def round_metric(val, metric_name):
    """Round accuracy/f1 to 4 decimals; EER: multiply by 100 then round to 2. Returns None if val is None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if metric_name == "eer":
        return round(float(val) * 100, 2)
    return round(float(val), 4)


def load_config(config_path):
    """Load config from JSON file. Returns None on failure."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        return None


def config_to_meta(config, run_dir_name=None):
    """
    Extract metadata from config for grouping and display.
    All keys are read from config; no parsing of directory names.

    Returns dict with:
        llm_path, llm_label, audio_encoder_type, audio_encoder_label,
        seed, freeze_audio_encoder, lora,
        use_weighted_layer_sum, weighted_layer_norm (if present),
        run_dir (optional)
    """
    if not config:
        return None

    model = config.get("model") or {}
    train = config.get("train") or {}

    llm_path = model.get("llm_path") or "unknown"
    # Short label: last part of path (e.g. Qwen2.5-0.5B-Instruct)
    llm_label = llm_path.split("/")[-1] if "/" in llm_path else llm_path

    audio_encoder_path = model.get("audio_encoder_path") or ""
    audio_encoder_type = model.get("audio_encoder_type") or "unknown"
    # Build rich audio_encoder_label: tokenizer configs (DAC, SpeechTokenizer) and default
    if audio_encoder_type == "dac":
        num_cb = model.get("dac_num_codebooks")
        discrete = model.get("dac_return_discrete", False)
        mode = "discrete" if discrete else "continuous"
        audio_encoder_label = f"dac_{num_cb}_{mode}" if num_cb is not None else f"dac_{mode}"
    elif audio_encoder_type == "speechtokenizer":
        num_layers = model.get("speechtokenizer_num_layers")
        audio_encoder_label = f"speechtokenizer_{num_layers}" if num_layers is not None else "speechtokenizer"
    elif audio_encoder_type != "unknown":
        audio_encoder_label = audio_encoder_type
    else:
        audio_encoder_label = (
            Path(audio_encoder_path).stem if audio_encoder_path else "unknown"
        )

    lora_rank = model.get("lora_rank", None)
    lora_alpha = model.get("lora_alpha", None)
    lora = model.get("lora", None)
    # When lora is used, append lora_rank and lora_alpha so different LoRA configs are separate
    if lora and (lora_rank is not None or lora_alpha is not None):
        parts = [str(lora_rank) if lora_rank is not None else "", str(lora_alpha) if lora_alpha is not None else ""]
        suffix = "_".join(p for p in parts if p)
        if suffix:
            audio_encoder_label = f"{audio_encoder_label}_lora{suffix}"

    seed = train.get("seed")
    if seed is not None:
        seed = str(seed)
    else:
        seed = "unknown"

    freeze_audio_encoder = model.get("freeze_audio_encoder", None)
    # Include tuned vs frozen in encoder label so results are grouped separately
    if freeze_audio_encoder is not None:
        audio_encoder_label = f"{audio_encoder_label}_frozen" if freeze_audio_encoder else f"{audio_encoder_label}_tuned"

    # Optional analysis_weights-specific keys
    use_weighted_layer_sum = model.get("use_weighted_layer_sum", None)
    weighted_layer_norm = model.get("weighted_layer_norm", None)

    meta = {
        "llm_path": llm_path,
        "llm_label": llm_label,
        "audio_encoder_type": audio_encoder_type,
        "audio_encoder_label": audio_encoder_label,
        "audio_encoder_path": audio_encoder_path,
        "seed": seed,
        "freeze_audio_encoder": freeze_audio_encoder,
        "lora": lora,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "use_weighted_layer_sum": use_weighted_layer_sum,
        "weighted_layer_norm": weighted_layer_norm,
    }
    if run_dir_name is not None:
        meta["run_dir"] = run_dir_name
    return meta


def load_metrics_json(json_path):
    """Load metrics from JSON. Expects keys: accuracy, f1_fake (or f1_score), eer."""
    try:
        with open(json_path, "r") as f:
            m = json.load(f)
        acc = m.get("accuracy")
        f1 = m.get("f1_fake") or m.get("f1_score")
        eer = m.get("eer")
        n = m.get("total_samples")
        return {
            "accuracy": float(acc) if acc is not None else None,
            "f1_score": float(f1) if f1 is not None else None,
            "eer": float(eer) if eer is not None else None,
            "num_samples": int(n) if n is not None else None,
        }
    except Exception as e:
        print(f"Error loading metrics {json_path}: {e}")
        return None


def normalize_dataset_name(dataset_name):
    """
    Unify dataset names so one logical dataset = one column.
    Eval outputs use: JSON NEW{dataset}_best_42NEW_metrics_best_42.json and
    CSV NEW{dataset}_best_42.csv. Strip trailing _best_42NEW, _best_42, _metrics_*
    so both map to the same name (e.g. NEWasv19_test).
    """
    if not dataset_name:
        return dataset_name
    name = dataset_name.strip()
    suffixes = ("_best_42NEW", "_best_42", "_metrics_best_42")
    while True:
        changed = False
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
                break
        if not changed:
            break
    return name or dataset_name


def extract_dataset_from_metrics_filename(filename, use_new=True, pattern_json=None):
    """
    Extract dataset name from metrics filename.
    use_new=True:  NEW{dataset}_best_42NEW_metrics_best_42.json -> NEW{dataset} (normalized).
    use_new=False: {dataset}_best_42_metrics_best_42.json -> {dataset}.
    """
    if not use_new:
        # Old naming: itw_best_42_metrics_best_42.json -> itw
        name = filename.replace(".json", "")
        if name.endswith("_best_42_metrics_best_42"):
            return name[: -len("_best_42_metrics_best_42")]
        return name or "unknown"
    # NEW naming: normalize so NEWasv19_test_best_42NEW -> NEWasv19_test
    m = re.match(r"^(.+?)_best_42NEW_metrics_best_42\.json$", filename, re.I)
    if m:
        return normalize_dataset_name(m.group(1))
    if pattern_json is None:
        pattern_json = re.compile(r"^(.+?)_(?:best_)?\d*_metrics_.*\.json$", re.I)
    match = pattern_json.match(filename)
    if match:
        return normalize_dataset_name(match.group(1))
    name = filename.replace(".json", "")
    for suffix in ["_metrics_best_42", "_best_42_metrics_best_42", "_best_42NEW_metrics_best_42"]:
        if suffix in name:
            name = name.split(suffix)[0]
            break
    return normalize_dataset_name(name) if name else "unknown"


def get_eer(y_true, y_score):
    """Compute EER from binary labels and scores."""
    fpr, fnr, _ = det_curve(y_true, y_score)
    index = np.nanargmin(np.absolute(fnr - fpr))
    return (fpr[index] + fnr[index]) / 2


def calculate_metrics_from_csv(csv_path):
    """
    Compute accuracy, F1, EER from CSV with ground_truth, prediction, score_real, score_fake.
    Returns dict with accuracy, f1_score, eer, num_samples or None.
    """
    try:
        df = pd.read_csv(csv_path)
        y_true = (df["ground_truth"].str.lower() == "real").astype(int).values
        y_pred = (df["prediction"].str.lower() == "real").astype(int).values
        y_scores = df["score_real"].values.astype(float) - df["score_fake"].values.astype(float)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred, average="macro")),
            "eer": float(get_eer(y_true, y_scores)),
            "num_samples": len(df),
        }
    except Exception as e:
        print(f"Error processing CSV {csv_path}: {e}")
        return None


def extract_dataset_from_csv_filename(filename, use_new=True):
    """use_new: NEWasv19_test_best_42.csv -> NEWasv19_test. Old: itw_best_42.csv -> itw."""
    name = filename.replace(".csv", "")
    if name.endswith("_best_42"):
        name = name[: -len("_best_42")]
    if use_new:
        return normalize_dataset_name(name) if name else name
    return name or "unknown"


def compute_pooled_from_run(run_path, use_new=True):
    """
    For one run (one seed): load best_42 CSVs, concatenate, compute accuracy/F1/EER.
    use_new=True:  only NEW*_best_42.csv. use_new=False: only *_best_42.csv that do not start with NEW.
    """
    run_path = Path(run_path)
    if use_new:
        csv_paths = [
            p for p in run_path.glob("NEW*_best_42.csv")
            if p.name.startswith("NEW") and "df21" not in str(p)
        ]
    else:
        csv_paths = [
            p for p in run_path.glob("*_best_42.csv")
            if not p.name.startswith("NEW") and "df21" not in str(p)
        ]
    if not csv_paths:
        return None
    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            for col in ("ground_truth", "prediction", "score_real", "score_fake"):
                if col not in df.columns:
                    break
            else:
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    y_true = (combined["ground_truth"].str.lower() == "real").astype(int).values
    y_pred = (combined["prediction"].str.lower() == "real").astype(int).values
    y_scores = combined["score_real"].values.astype(float) - combined["score_fake"].values.astype(float)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="macro")),
        "eer": float(get_eer(y_true, y_scores)),
    }


def compute_pooled_weighted_from_metrics_list(metrics_list):
    """
    Fallback when run has no CSVs: one pooled value per seed = weighted average
    of per-dataset metrics by num_samples (same as would be if we had concatenated).
    metrics_list: list of (dataset_name, metrics_dict, file_path).
    """
    sum_acc, n_acc = 0.0, 0
    sum_f1, n_f1 = 0.0, 0
    sum_eer, n_eer = 0.0, 0
    for _, metrics, _ in metrics_list:
        n = metrics.get("num_samples") or 1
        if n <= 0:
            n = 1
        a = metrics.get("accuracy")
        f = metrics.get("f1_score")
        e = metrics.get("eer")
        if a is not None:
            sum_acc += a * n
            n_acc += n
        if f is not None:
            sum_f1 += f * n
            n_f1 += n
        if e is not None:
            sum_eer += e * n
            n_eer += n
    return {
        "accuracy": sum_acc / n_acc if n_acc else None,
        "f1_score": sum_f1 / n_f1 if n_f1 else None,
        "eer": sum_eer / n_eer if n_eer else None,
    }


def find_metrics_in_run(run_dir, use_new=True):
    """
    Find all metrics in a run directory.
    use_new=True:  only NEW*_metrics_best_42.json and NEW*_best_42.csv.
    use_new=False: only *best_42_metrics_best_42.json and *_best_42.csv that do NOT start with NEW.
    Returns list of (dataset_name, metrics_dict, file_path).
    """
    run_path = Path(run_dir)
    results = []
    seen_datasets = set()

    if use_new:
        json_glob = "NEW*_metrics_best_42.json"
        csv_glob = "NEW*_best_42.csv"
        def json_ok(p):
            return p.name.startswith("NEW") and "df21" not in str(p)
        def csv_ok(p):
            return p.name.startswith("NEW") and "df21" not in str(p)
    else:
        json_glob = "*best_42_metrics_best_42.json"
        csv_glob = "*_best_42.csv"
        def json_ok(p):
            return not p.name.startswith("NEW") and "df21" not in str(p)
        def csv_ok(p):
            return not p.name.startswith("NEW") and "df21" not in str(p)

    for json_path in run_path.glob(json_glob):
        if not json_ok(json_path):
            continue
        metrics = load_metrics_json(json_path)
        if metrics is None:
            continue
        dataset = extract_dataset_from_metrics_filename(json_path.name, use_new=use_new)
        results.append((dataset, metrics, str(json_path)))
        seen_datasets.add(dataset)

    for csv_path in run_path.glob(csv_glob):
        if not csv_ok(csv_path):
            continue
        dataset = extract_dataset_from_csv_filename(csv_path.name, use_new=use_new)
        if dataset in seen_datasets:
            continue
        metrics = calculate_metrics_from_csv(csv_path)
        if metrics is not None:
            results.append((dataset, metrics, str(csv_path)))
    return results


def find_run_dirs_with_config(base_dir):
    """
    Find all run directories under base_dir that contain config.json.
    Returns list of (run_dir Path, config dict, run_dir_name).
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Warning: Directory {base_dir} does not exist")
        return []

    runs = []
    for run_path in base_path.iterdir():
        if not run_path.is_dir():
            continue
        config_path = run_path / "config.json"
        if not config_path.exists():
            continue
        config = load_config(config_path)
        if config is None:
            continue
        runs.append((run_path, config, run_path.name))
    return runs


def organize_results(all_results):
    """
    Organize by llm_label, audio_encoder_label, seed.
    Structure: organized[llm_label][audio_encoder_label][seed] = list of {dataset, metrics, ...}
    """
    organized = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for r in all_results:
        if r is None:
            continue
        llm = r["llm_label"]
        enc = r["audio_encoder_label"]
        seed = r["seed"]
        organized[llm][enc][seed].append({
            "dataset": r["dataset"],
            "accuracy": r["accuracy"],
            "f1_score": r["f1_score"],
            "eer": r["eer"],
            "num_samples": r.get("num_samples"),
            "metrics_path": r.get("metrics_path"),
            "freeze_audio_encoder": r.get("freeze_audio_encoder"),
        })
    return organized


def main():
    parser = argparse.ArgumentParser(description="Analyze analysis_weights results from config-driven run dirs.")
    parser.add_argument(
        "--base_dirs",
        type=str,
        nargs="+",
        default=["/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/analysis_weights"],
        help="One or more base directories containing run subdirs with config.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs",
        help="Directory for output CSV/JSON files",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="analysis_weights",
        help="Prefix for output filenames (e.g. name_summary.csv, name_organized.json)",
    )
    parser.add_argument(
        "--metrics_mode",
        type=str,
        choices=["new", "old"],
        default="old",
        help="Metric files: 'new' = only NEW* files (NEW{dataset}_best_42NEW_metrics...). "
             "'old' = only non-NEW files ({dataset}_best_42_metrics_best_42.json). Use 'old' for folders with classic best_42 naming.",
    )
    args = parser.parse_args()
    base_dirs = args.base_dirs
    out_dir = Path(args.output_dir)
    name = args.name.strip() or "analysis_weights"
    use_new = args.metrics_mode == "new"

    print(f"Metrics mode: {args.metrics_mode} (use_new={use_new})")
    print("Finding run directories with config.json...")
    runs = []
    for base_dir in base_dirs:
        r = find_run_dirs_with_config(base_dir)
        runs.extend(r)
        if r:
            print(f"  {base_dir}: {len(r)} runs")
    print(f"Total: {len(runs)} run directories")

    all_results = []
    run_pooled = {}  # (llm_label, audio_encoder_label, seed) -> {accuracy, f1_score, eer} (one per seed = concat all datasets)
    for run_path, config, run_name in runs:
        meta = config_to_meta(config, run_dir_name=run_name)
        if meta is None:
            continue

        metrics_list = find_metrics_in_run(run_path, use_new=use_new)
        if not metrics_list:
            print(f"No metrics in: {run_name}")
            continue

        # Pooled per seed: concatenate all datasets (from CSV) or weighted avg (from JSON only)
        pooled = compute_pooled_from_run(run_path, use_new=use_new)
        if pooled is None:
            pooled = compute_pooled_weighted_from_metrics_list(metrics_list)
        if pooled:
            key = (meta["llm_label"], meta["audio_encoder_label"], meta["seed"])
            run_pooled[key] = pooled

        for dataset, metrics, metrics_path in metrics_list:
            all_results.append({
                **meta,
                "dataset": dataset,
                "accuracy": metrics.get("accuracy"),
                "f1_score": metrics.get("f1_score"),
                "eer": metrics.get("eer"),
                "num_samples": metrics.get("num_samples"),
                "metrics_path": metrics_path,
            })
        print(f"Processed: {run_name} ({len(metrics_list)} metric files)")

    print(f"\nTotal result rows: {len(all_results)}")
    if not all_results:
        print("No results to analyze.")
        return

    organized = organize_results(all_results)

    # Flatten for summary table
    summary_data = []
    for r in all_results:
        summary_data.append({
            "llm_label": r["llm_label"],
            "audio_encoder": r["audio_encoder_label"],
            "seed": r["seed"],
            "freeze_audio_encoder": r.get("freeze_audio_encoder"),
            "lora": r.get("lora"),
            "lora_rank": r.get("lora_rank"),
            "lora_alpha": r.get("lora_alpha"),
            "use_weighted_layer_sum": r.get("use_weighted_layer_sum"),
            "weighted_layer_norm": r.get("weighted_layer_norm"),
            "run_dir": r.get("run_dir"),
            "dataset": r["dataset"],
            "accuracy": round_metric(r["accuracy"], "accuracy"),
            "f1_score": round_metric(r["f1_score"], "f1_score"),
            "eer": round_metric(r["eer"], "eer"),
            "num_samples": r.get("num_samples"),
        })

    df_summary = pd.DataFrame(summary_data)
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("\nDetailed Results:")
    print(df_summary.to_string(index=False))

    summary_csv = out_dir / f"{name}_summary.csv"
    df_summary.to_csv(summary_csv, index=False, sep="\t")
    print(f"\nSummary saved to: {summary_csv}")

    # Averages per (llm, encoder, seed) across datasets
    avg_per_seed_data = []
    for llm in sorted(organized.keys()):
        for enc in sorted(organized[llm].keys()):
            for seed in sorted(organized[llm][enc].keys()):
                rows = organized[llm][enc][seed]
                acc = [x["accuracy"] for x in rows if x["accuracy"] is not None]
                f1 = [x["f1_score"] for x in rows if x["f1_score"] is not None]
                eer = [x["eer"] for x in rows if x["eer"] is not None]
                if not acc:
                    continue
                freeze = rows[0].get("freeze_audio_encoder") if rows else None
                avg_per_seed_data.append({
                    "llm_label": llm,
                    "audio_encoder": enc,
                    "freeze_audio_encoder": freeze,
                    "seed": seed,
                    "num_datasets": len(rows),
                    "avg_accuracy": round_metric(float(np.mean(acc)), "accuracy"),
                    "std_accuracy": round_metric(float(np.std(acc)) if len(acc) > 1 else 0.0, "accuracy"),
                    "avg_f1_score": round_metric(float(np.mean(f1)) if f1 else None, "f1_score"),
                    "std_f1_score": round_metric(float(np.std(f1)) if f1 and len(f1) > 1 else None, "f1_score"),
                    "avg_eer": round_metric(float(np.mean(eer)) if eer else None, "eer"),
                    "std_eer": round_metric(float(np.std(eer)) if eer and len(eer) > 1 else None, "eer"),
                })

    if avg_per_seed_data:
        df_avg_seed = pd.DataFrame(avg_per_seed_data)
        print("\n" + "=" * 80)
        print("AVERAGE ACROSS DATASETS (per run / seed)")
        print("=" * 80)
        print(df_avg_seed.to_string(index=False))
        avg_seed_csv = out_dir / f"{name}_averaged_per_seed.csv"
        df_avg_seed.to_csv(avg_seed_csv, index=False)
        print(f"\nSaved: {avg_seed_csv}")

    # Averages across seeds per (llm, encoder)
    avg_across_seeds_data = []
    for llm in sorted(organized.keys()):
        for enc in sorted(organized[llm].keys()):
            seeds_data = []
            for seed in sorted(organized[llm][enc].keys()):
                rows = organized[llm][enc][seed]
                acc = [x["accuracy"] for x in rows if x["accuracy"] is not None]
                f1 = [x["f1_score"] for x in rows if x["f1_score"] is not None]
                eer = [x["eer"] for x in rows if x["eer"] is not None]
                if acc:
                    seeds_data.append({
                        "seed": seed,
                        "avg_accuracy": np.mean(acc),
                        "avg_f1_score": np.mean(f1) if f1 else None,
                        "avg_eer": np.mean(eer) if eer else None,
                    })
            if not seeds_data:
                continue
            accs = [s["avg_accuracy"] for s in seeds_data]
            f1s = [s["avg_f1_score"] for s in seeds_data if s["avg_f1_score"] is not None]
            eers = [s["avg_eer"] for s in seeds_data if s["avg_eer"] is not None]
            first_rows = organized[llm][enc][next(iter(organized[llm][enc].keys()))]
            freeze = first_rows[0].get("freeze_audio_encoder") if first_rows else None
            avg_across_seeds_data.append({
                "llm_label": llm,
                "audio_encoder": enc,
                "freeze_audio_encoder": freeze,
                "num_seeds": len(seeds_data),
                "seeds": ", ".join(sorted(s["seed"] for s in seeds_data)),
                "avg_accuracy": round_metric(float(np.mean(accs)), "accuracy"),
                "std_accuracy": round_metric(float(np.std(accs)) if len(accs) > 1 else 0.0, "accuracy"),
                "avg_f1_score": round_metric(float(np.mean(f1s)) if f1s else None, "f1_score"),
                "std_f1_score": round_metric(float(np.std(f1s)) if len(f1s) > 1 else None, "f1_score"),
                "avg_eer": round_metric(float(np.mean(eers)) if eers else None, "eer"),
                "std_eer": round_metric(float(np.std(eers)) if len(eers) > 1 else None, "eer"),
            })

    if avg_across_seeds_data:
        df_avg_seeds = pd.DataFrame(avg_across_seeds_data)
        print("\n" + "=" * 80)
        print("AVERAGE ACROSS SEEDS (per llm + audio_encoder)")
        print("=" * 80)
        print(df_avg_seeds.to_string(index=False))
        avg_seeds_csv = out_dir / f"{name}_averaged_across_seeds.csv"
        df_avg_seeds.to_csv(avg_seeds_csv, index=False, sep="\t")
        print(f"\nSaved: {avg_seeds_csv}")

    # --- Per-dataset averages table: one row per (llm, encoder), columns = [llm_label, audio_encoder, num_seeds, seeds] + for each dataset: avg_accuracy, avg_f1_score, avg_eer ---
    all_datasets = sorted({r["dataset"] for r in all_results})
    if all_datasets:
        # (llm, enc) -> dataset -> list of (acc, f1, eer) across seeds
        by_config = defaultdict(lambda: defaultdict(list))
        for llm in organized.keys():
            for enc in organized[llm].keys():
                for seed in organized[llm][enc].keys():
                    if seed == "_avg":
                        continue
                    for row in organized[llm][enc][seed]:
                        ds = row["dataset"]
                        by_config[(llm, enc)][ds].append({
                            "accuracy": row["accuracy"],
                            "f1_score": row["f1_score"],
                            "eer": row["eer"],
                            "num_samples": row.get("num_samples"),
                        })

        rows_avg = []
        rows_std = []
        for llm in sorted(organized.keys()):
            for enc in sorted(organized[llm].keys()):
                seeds_set = {s for s in organized[llm][enc].keys() if s != "_avg"}
                num_seeds = len(seeds_set)
                seeds_str = ", ".join(sorted(seeds_set))
                first_seed = next(iter(seeds_set))
                first_rows = organized[llm][enc][first_seed]
                freeze = first_rows[0].get("freeze_audio_encoder") if first_rows else None
                row_avg = {
                    "llm_label": llm,
                    "audio_encoder": enc,
                    "freeze_audio_encoder": freeze,
                    "num_seeds": num_seeds,
                    "seeds": seeds_str,
                }
                row_std = {
                    "llm_label": llm,
                    "audio_encoder": enc,
                    "freeze_audio_encoder": freeze,
                    "num_seeds": num_seeds,
                    "seeds": seeds_str,
                }
                for ds in all_datasets:
                    vals = by_config[(llm, enc)].get(ds, [])
                    acc = [v["accuracy"] for v in vals if v["accuracy"] is not None]
                    f1 = [v["f1_score"] for v in vals if v["f1_score"] is not None]
                    eer = [v["eer"] for v in vals if v["eer"] is not None]
                    # Average table (rounded)
                    row_avg[f"avg_accuracy_{ds}"] = round_metric(float(np.mean(acc)) if acc else None, "accuracy")
                    row_avg[f"avg_f1_score_{ds}"] = round_metric(float(np.mean(f1)) if f1 else None, "f1_score")
                    row_avg[f"avg_eer_{ds}"] = round_metric(float(np.mean(eer)) if eer else None, "eer")
                    # Std table (rounded)
                    row_std[f"std_accuracy_{ds}"] = round_metric(float(np.std(acc)) if len(acc) > 1 else (0.0 if acc else None), "accuracy")
                    row_std[f"std_f1_score_{ds}"] = round_metric(float(np.std(f1)) if len(f1) > 1 else (0.0 if f1 else None), "f1_score")
                    row_std[f"std_eer_{ds}"] = round_metric(float(np.std(eer)) if len(eer) > 1 else (0.0 if eer else None), "eer")
                # Last columns: average across datasets per metric (rounded)
                def _safe_nanmean(vals):
                    f = [float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else np.nan for x in vals]
                    return float(np.nanmean(f)) if not np.all(np.isnan(f)) else None
                acc_avgs = [row_avg[f"avg_accuracy_{ds}"] for ds in all_datasets]
                f1_avgs = [row_avg[f"avg_f1_score_{ds}"] for ds in all_datasets]
                eer_avgs = [row_avg[f"avg_eer_{ds}"] for ds in all_datasets]
                row_avg["avg_accuracy_mean"] = round_metric(_safe_nanmean(acc_avgs), "accuracy")
                row_avg["avg_f1_score_mean"] = round_metric(_safe_nanmean(f1_avgs), "f1_score")
                # Average EER: mean of per-dataset EERs (already in %), round to 2, no extra ×100
                row_avg["avg_eer_mean"] = round(_safe_nanmean(eer_avgs), 2) if _safe_nanmean(eer_avgs) is not None else None
                # Pooled average: per seed = metric on concatenated datasets; then average those values across seeds
                pool_per_seed = [run_pooled.get((llm, enc, seed)) for seed in seeds_set]
                pool_per_seed = [p for p in pool_per_seed if p is not None]
                if pool_per_seed:
                    accs = [p["accuracy"] for p in pool_per_seed if p.get("accuracy") is not None]
                    f1s = [p["f1_score"] for p in pool_per_seed if p.get("f1_score") is not None]
                    eers = [p["eer"] for p in pool_per_seed if p.get("eer") is not None]
                    row_avg["pooled_accuracy"] = round_metric(float(np.mean(accs)), "accuracy") if accs else None
                    row_avg["pooled_f1_score"] = round_metric(float(np.mean(f1s)), "f1_score") if f1s else None
                    row_avg["pooled_eer"] = round_metric(float(np.mean(eers)), "eer") if eers else None
                else:
                    row_avg["pooled_accuracy"] = None
                    row_avg["pooled_f1_score"] = None
                    row_avg["pooled_eer"] = None
                acc_stds = [row_std[f"std_accuracy_{ds}"] for ds in all_datasets]
                f1_stds = [row_std[f"std_f1_score_{ds}"] for ds in all_datasets]
                eer_stds = [row_std[f"std_eer_{ds}"] for ds in all_datasets]
                row_std["std_accuracy_mean"] = round_metric(_safe_nanmean(acc_stds), "accuracy")
                row_std["std_f1_score_mean"] = round_metric(_safe_nanmean(f1_stds), "f1_score")
                # Std EER mean: mean of per-dataset EER stds, round to 2, no ×100
                row_std["std_eer_mean"] = round(_safe_nanmean(eer_stds), 2) if _safe_nanmean(eer_stds) is not None else None
                rows_avg.append(row_avg)
                rows_std.append(row_std)

        df_per_dataset_avg = pd.DataFrame(rows_avg)
        df_per_dataset_std = pd.DataFrame(rows_std)
        print("\n" + "=" * 80)
        print("PER-DATASET AVERAGES (one row per llm + audio_encoder, columns = avg per dataset)")
        print("=" * 80)
        print(df_per_dataset_avg.to_string(index=False))
        per_ds_avg_csv = out_dir / f"{name}_per_dataset_avg.csv"
        df_per_dataset_avg.to_csv(per_ds_avg_csv, index=False, sep="\t")
        print(f"\nSaved: {per_ds_avg_csv}")

        print("\n" + "=" * 80)
        print("PER-DATASET STD (one row per llm + audio_encoder, columns = std per dataset)")
        print("=" * 80)
        print(df_per_dataset_std.to_string(index=False))
        per_ds_std_csv = out_dir / f"{name}_per_dataset_std.csv"
        df_per_dataset_std.to_csv(per_ds_std_csv, index=False, sep="\t")
        print(f"\nSaved: {per_ds_std_csv}")

    # Add per-dataset averages across seeds to organized structure
    for llm in organized.keys():
        for enc in organized[llm].keys():
            by_dataset = defaultdict(list)
            for seed in organized[llm][enc].keys():
                for row in organized[llm][enc][seed]:
                    by_dataset[row["dataset"]].append(row)
            avg_by_dataset = []
            for ds, rows in sorted(by_dataset.items()):
                acc = [x["accuracy"] for x in rows if x["accuracy"] is not None]
                f1 = [x["f1_score"] for x in rows if x["f1_score"] is not None]
                eer = [x["eer"] for x in rows if x["eer"] is not None]
                avg_by_dataset.append({
                    "dataset": ds,
                    "avg_accuracy": float(np.mean(acc)) if acc else None,
                    "std_accuracy": float(np.std(acc)) if len(acc) > 1 else None,
                    "avg_f1_score": float(np.mean(f1)) if f1 else None,
                    "std_f1_score": float(np.std(f1)) if len(f1) > 1 else None,
                    "avg_eer": float(np.mean(eer)) if eer else None,
                    "std_eer": float(np.std(eer)) if len(eer) > 1 else None,
                    "num_seeds": len(rows),
                })
            organized[llm][enc]["_avg"] = avg_by_dataset

    def to_serializable(obj):
        if isinstance(obj, defaultdict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        return obj

    organized_plain = to_serializable(organized)
    out_json = out_dir / f"{name}_organized.json"
    with open(out_json, "w") as f:
        json.dump(organized_plain, f, indent=2)
    print(f"\nOrganized results saved to: {out_json}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
