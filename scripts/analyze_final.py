#!/usr/bin/env python3
"""
Final analysis script: prefer NEW metrics over OLD per dataset; ignore mlaad_en only.
For datasets that have both NEW and OLD files, use only NEW. If NEW is missing, use OLD.
Always exclude dataset mlaad_en (other mlaad_* e.g. mlaad_all are kept in one variant).
Only metric files whose filename contains "42" are used (e.g. *_best_42_metrics_best_42.json,
*_best_42.csv); others (e.g. asv19_test_best_metrics_best.json) are ignored.
Runs (folders) that do not have metric files for all datasets are excluded so they do not affect averages.

Outputs four sets of CSVs:
  - analyze_final_*.csv  : all datasets (including mlaad_all), excluding mlaad_en
  - analyze_final_womlaad_*.csv : same but excluding mlaad_all as well (without mlaad_all)
  - analyze_final_la19_itw_la21_*.csv : only la19 (asv19_test), itw, la21; includes pooled
  - analyze_final_la19_itw_la21_mlaad_*.csv : la19, itw, la21 + mlaad_all; includes pooled
  - analyze_final_la19_itw_la21_mlaad_en_*.csv : la19, itw, la21 + mlaad_en; includes pooled
"""

import argparse
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import det_curve, f1_score, accuracy_score

# Dataset to always exclude
EXCLUDE_DATASET_MLAAD_EN = "mlaad_en"
# Dataset to exclude only in "womlaad" variant
EXCLUDE_DATASET_MLAAD_ALL = "mlaad_all"
# Datasets for the "la19_itw_la21" variant (la19 = asv19_test or la19)
DATASETS_LA19_ITW_LA21 = {"asv19_test", "la19", "itw", "la21"}
# Same three + mlaad_all
DATASETS_LA19_ITW_LA21_MLAAD = DATASETS_LA19_ITW_LA21 | {EXCLUDE_DATASET_MLAAD_ALL}
# Same three + mlaad_en
DATASETS_LA19_ITW_LA21_MLAAD_EN = DATASETS_LA19_ITW_LA21 | {EXCLUDE_DATASET_MLAAD_EN}


def round_metric(val, metric_name):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if metric_name == "eer":
        return round(float(val) * 100, 2)
    return round(float(val), 4)


def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        return None


def config_to_meta(config, run_dir_name=None):
    if not config:
        return None
    model = config.get("model") or {}
    train = config.get("train") or {}
    llm_path = model.get("llm_path") or "unknown"
    llm_label = llm_path.split("/")[-1] if "/" in llm_path else llm_path
    audio_encoder_path = model.get("audio_encoder_path") or ""
    audio_encoder_type = model.get("audio_encoder_type") or "unknown"
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
        audio_encoder_label = Path(audio_encoder_path).stem if audio_encoder_path else "unknown"
    lora_rank = model.get("lora_rank", None)
    lora_alpha = model.get("lora_alpha", None)
    lora = model.get("lora", None)
    if lora and (lora_rank is not None or lora_alpha is not None):
        parts = [str(lora_rank) if lora_rank is not None else "", str(lora_alpha) if lora_alpha is not None else ""]
        suffix = "_".join(p for p in parts if p)
        if suffix:
            audio_encoder_label = f"{audio_encoder_label}_lora{suffix}"
    seed = train.get("seed")
    seed = str(seed) if seed is not None else "unknown"
    freeze_audio_encoder = model.get("freeze_audio_encoder", None)
    if freeze_audio_encoder is not None:
        audio_encoder_label = f"{audio_encoder_label}_frozen" if freeze_audio_encoder else f"{audio_encoder_label}_tuned"
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
        "use_weighted_layer_sum": model.get("use_weighted_layer_sum", None),
        "weighted_layer_norm": model.get("weighted_layer_norm", None),
    }
    if run_dir_name is not None:
        meta["run_dir"] = run_dir_name
    return meta


def load_metrics_json(json_path):
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
    if not dataset_name:
        return dataset_name
    name = dataset_name.strip()
    for suffix in ("_best_42NEW", "_best_42", "_metrics_best_42"):
        while name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or dataset_name


def canonical_dataset(name):
    """One key per logical dataset: NEWasv19_test and asv19_test -> asv19_test."""
    n = normalize_dataset_name(name)
    return n[3:] if n.startswith("NEW") else n


def extract_dataset_from_json_filename_new(filename):
    m = re.match(r"^NEW(.+?)_best_42NEW_metrics_best_42\.json$", filename, re.I)
    if m:
        return "NEW" + normalize_dataset_name(m.group(1))
    if filename.startswith("NEW") and "_metrics_best_42.json" in filename:
        name = filename.replace("_metrics_best_42.json", "").replace("_best_42NEW", "").strip()
        return normalize_dataset_name(name) if name else "unknown"
    return None


def extract_dataset_from_json_filename_old(filename):
    if filename.endswith("_best_42_metrics_best_42.json"):
        return filename[: -len("_best_42_metrics_best_42.json")]
    return None


def extract_dataset_from_csv_filename_new(filename):
    if not filename.startswith("NEW") or not filename.endswith("_best_42.csv"):
        return None
    return normalize_dataset_name(filename[: -len("_best_42.csv")])


def extract_dataset_from_csv_filename_old(filename):
    if not filename.endswith("_best_42.csv") or filename.startswith("NEW"):
        return None
    return filename[: -len("_best_42.csv")]


def get_eer(y_true, y_score):
    fpr, fnr, _ = det_curve(y_true, y_score)
    index = np.nanargmin(np.absolute(fnr - fpr))
    return (fpr[index] + fnr[index]) / 2


def calculate_metrics_from_csv(csv_path):
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


def compute_pooled_weighted_from_metrics_list(metrics_list):
    sum_acc, n_acc = 0.0, 0
    sum_f1, n_f1 = 0.0, 0
    sum_eer, n_eer = 0.0, 0
    for _, metrics, _ in metrics_list:
        n = metrics.get("num_samples") or 1
        n = max(1, n)
        a, f, e = metrics.get("accuracy"), metrics.get("f1_score"), metrics.get("eer")
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


def find_metrics_in_run_final(run_dir):
    """
    Per dataset: use NEW file if present, else OLD. Collect all datasets including mlaad_en
    (mlaad_en is excluded later from main outputs but used in la19_itw_la21_mlaad_en variant).
    Only consider JSON/CSV files whose filename contains "42" (e.g. best_42, metrics_best_42).
    Returns list of (canonical_dataset_name, metrics_dict, file_path).
    """
    run_path = Path(run_dir)
    # canonical -> ("new"|"old", canonical_name, metrics_dict, file_path)
    chosen = {}

    def _has_42(path):
        """Only use files with '42' in the filename (e.g. best_42, metrics_best_42)."""
        return "42" in path.name

    # 1) NEW JSON
    for json_path in run_path.glob("NEW*_metrics_best_42.json"):
        if not _has_42(json_path) or "df21" in str(json_path):
            continue
        name = extract_dataset_from_json_filename_new(json_path.name)
        if not name:
            continue
        can = canonical_dataset(name)
        if can not in chosen:
            metrics = load_metrics_json(json_path)
            if metrics is not None:
                chosen[can] = ("new", can, metrics, str(json_path))

    # 2) NEW CSV (only for canonical not already from NEW json)
    for csv_path in run_path.glob("NEW*_best_42.csv"):
        if not _has_42(csv_path) or "df21" in str(csv_path):
            continue
        name = extract_dataset_from_csv_filename_new(csv_path.name)
        if not name:
            continue
        can = canonical_dataset(name)
        if can not in chosen:
            metrics = calculate_metrics_from_csv(csv_path)
            if metrics is not None:
                chosen[can] = ("new", can, metrics, str(csv_path))

    # 3) OLD JSON
    for json_path in run_path.glob("*best_42_metrics_best_42.json"):
        if not _has_42(json_path) or json_path.name.startswith("NEW"):
            continue
        if "df21" in str(json_path):
            continue
        name = extract_dataset_from_json_filename_old(json_path.name)
        if not name:
            continue
        can = canonical_dataset(name)
        if can not in chosen:
            metrics = load_metrics_json(json_path)
            if metrics is not None:
                chosen[can] = ("old", can, metrics, str(json_path))

    # 4) OLD CSV
    for csv_path in run_path.glob("*_best_42.csv"):
        if not _has_42(csv_path) or csv_path.name.startswith("NEW") or "df21" in str(csv_path):
            continue
        name = extract_dataset_from_csv_filename_old(csv_path.name)
        if not name:
            continue
        can = canonical_dataset(name)
        if can not in chosen:
            metrics = calculate_metrics_from_csv(csv_path)
            if metrics is not None:
                chosen[can] = ("old", can, metrics, str(csv_path))

    return [(can, m, p) for _s, can, m, p in chosen.values()]


def _filter_out_mlaad_en(results):
    """Exclude mlaad_en rows (for main analyze_final and womlaad outputs)."""
    return [r for r in results if r["dataset"] != EXCLUDE_DATASET_MLAAD_EN]


def find_run_dirs_with_config(base_dir):
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
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in all_results:
        if r is None:
            continue
        llm, enc, seed = r["llm_label"], r["audio_encoder_label"], r["seed"]
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


def run_analysis(all_results, run_pooled, out_dir, name_prefix):
    """Build summary, averaged per seed, across seeds, per-dataset avg/std; save under out_dir with name_prefix."""
    if not all_results:
        return
    organized = organize_results(all_results)
    out_dir = Path(out_dir)

    # Summary table
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
            "run_dir": r.get("run_dir"),
            "dataset": r["dataset"],
            "accuracy": round_metric(r["accuracy"], "accuracy"),
            "f1_score": round_metric(r["f1_score"], "f1_score"),
            "eer": round_metric(r["eer"], "eer"),
            "num_samples": r.get("num_samples"),
        })
    df_summary = pd.DataFrame(summary_data)
    summary_csv = out_dir / f"{name_prefix}_summary.csv"
    df_summary.to_csv(summary_csv, index=False, sep="\t")
    print(f"Saved: {summary_csv}")

    # Average per seed
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
        avg_seed_csv = out_dir / f"{name_prefix}_averaged_per_seed.csv"
        df_avg_seed.to_csv(avg_seed_csv, index=False)
        print(f"Saved: {avg_seed_csv}")

    # Average across seeds
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
        avg_seeds_csv = out_dir / f"{name_prefix}_averaged_across_seeds.csv"
        df_avg_seeds.to_csv(avg_seeds_csv, index=False, sep="\t")
        print(f"Saved: {avg_seeds_csv}")

    # Per-dataset avg/std
    all_datasets = sorted({r["dataset"] for r in all_results})
    if not all_datasets:
        return
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

    rows_avg, rows_std = [], []
    for llm in sorted(organized.keys()):
        for enc in sorted(organized[llm].keys()):
            seeds_set = {s for s in organized[llm][enc].keys() if s != "_avg"}
            num_seeds = len(seeds_set)
            seeds_str = ", ".join(sorted(seeds_set))
            first_seed = next(iter(seeds_set))
            first_rows = organized[llm][enc][first_seed]
            freeze = first_rows[0].get("freeze_audio_encoder") if first_rows else None
            row_avg = {"llm_label": llm, "audio_encoder": enc, "freeze_audio_encoder": freeze, "num_seeds": num_seeds, "seeds": seeds_str}
            row_std = {"llm_label": llm, "audio_encoder": enc, "freeze_audio_encoder": freeze, "num_seeds": num_seeds, "seeds": seeds_str}
            for ds in all_datasets:
                vals = by_config[(llm, enc)].get(ds, [])
                acc = [v["accuracy"] for v in vals if v["accuracy"] is not None]
                f1 = [v["f1_score"] for v in vals if v["f1_score"] is not None]
                eer = [v["eer"] for v in vals if v["eer"] is not None]
                row_avg[f"avg_accuracy_{ds}"] = round_metric(float(np.mean(acc)) if acc else None, "accuracy")
                row_avg[f"avg_f1_score_{ds}"] = round_metric(float(np.mean(f1)) if f1 else None, "f1_score")
                row_avg[f"avg_eer_{ds}"] = round_metric(float(np.mean(eer)) if eer else None, "eer")
                row_std[f"std_accuracy_{ds}"] = round_metric(float(np.std(acc)) if len(acc) > 1 else (0.0 if acc else None), "accuracy")
                row_std[f"std_f1_score_{ds}"] = round_metric(float(np.std(f1)) if len(f1) > 1 else (0.0 if f1 else None), "f1_score")
                row_std[f"std_eer_{ds}"] = round_metric(float(np.std(eer)) if len(eer) > 1 else (0.0 if eer else None), "eer")
            def _safe_nanmean(vals):
                f = [float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else np.nan for x in vals]
                return float(np.nanmean(f)) if not np.all(np.isnan(f)) else None
            acc_avgs = [row_avg[f"avg_accuracy_{ds}"] for ds in all_datasets]
            f1_avgs = [row_avg[f"avg_f1_score_{ds}"] for ds in all_datasets]
            eer_avgs = [row_avg[f"avg_eer_{ds}"] for ds in all_datasets]
            row_avg["avg_accuracy_mean"] = round_metric(_safe_nanmean(acc_avgs), "accuracy")
            row_avg["avg_f1_score_mean"] = round_metric(_safe_nanmean(f1_avgs), "f1_score")
            row_avg["avg_eer_mean"] = round(_safe_nanmean(eer_avgs), 2) if _safe_nanmean(eer_avgs) is not None else None
            key_list = [(llm, enc, seed) for seed in seeds_set]
            pool_per_seed = [run_pooled.get(k) for k in key_list]
            pool_per_seed = [p for p in pool_per_seed if p is not None]
            if pool_per_seed:
                accs = [p["accuracy"] for p in pool_per_seed if p.get("accuracy") is not None]
                f1s = [p["f1_score"] for p in pool_per_seed if p.get("f1_score") is not None]
                eers = [p["eer"] for p in pool_per_seed if p.get("eer") is not None]
                row_avg["pooled_accuracy"] = round_metric(float(np.mean(accs)), "accuracy") if accs else None
                row_avg["pooled_f1_score"] = round_metric(float(np.mean(f1s)), "f1_score") if f1s else None
                row_avg["pooled_eer"] = round_metric(float(np.mean(eers)), "eer") if eers else None
            else:
                row_avg["pooled_accuracy"] = row_avg["pooled_f1_score"] = row_avg["pooled_eer"] = None
            acc_stds = [row_std[f"std_accuracy_{ds}"] for ds in all_datasets]
            f1_stds = [row_std[f"std_f1_score_{ds}"] for ds in all_datasets]
            eer_stds = [row_std[f"std_eer_{ds}"] for ds in all_datasets]
            row_std["std_accuracy_mean"] = round_metric(_safe_nanmean(acc_stds), "accuracy")
            row_std["std_f1_score_mean"] = round_metric(_safe_nanmean(f1_stds), "f1_score")
            row_std["std_eer_mean"] = round(_safe_nanmean(eer_stds), 2) if _safe_nanmean(eer_stds) is not None else None
            rows_avg.append(row_avg)
            rows_std.append(row_std)

    df_per_avg = pd.DataFrame(rows_avg)
    df_per_std = pd.DataFrame(rows_std)
    per_avg_csv = out_dir / f"{name_prefix}_per_dataset_avg.csv"
    per_std_csv = out_dir / f"{name_prefix}_per_dataset_std.csv"
    df_per_avg.to_csv(per_avg_csv, index=False, sep="\t")
    df_per_std.to_csv(per_std_csv, index=False, sep="\t")
    print(f"Saved: {per_avg_csv}")
    print(f"Saved: {per_std_csv}")


def main():
    parser = argparse.ArgumentParser(description="Final analysis: prefer NEW over OLD per dataset, ignore mlaad_en; output with and without mlaad_all.")
    parser.add_argument(
        "--base_dirs",
        type=str,
        nargs="+",
        default=["/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/analysis_weights"],
        help="Base directories containing run subdirs with config.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs",
        help="Directory for output CSV files",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Finding run directories with config.json...")
    runs = []
    for base_dir in args.base_dirs:
        r = find_run_dirs_with_config(base_dir)
        runs.extend(r)
        if r:
            print(f"  {base_dir}: {len(r)} runs")
    print(f"Total: {len(runs)} run directories")
    print(f"Rules: exclude {EXCLUDE_DATASET_MLAAD_EN}; prefer NEW over OLD per dataset")

    all_results = []
    run_pooled = {}
    for run_path, config, run_name in runs:
        meta = config_to_meta(config, run_dir_name=run_name)
        if meta is None:
            continue
        metrics_list = find_metrics_in_run_final(run_path)
        if not metrics_list:
            print(f"No metrics in: {run_name}")
            continue
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

    print(f"\nTotal result rows (all datasets, including {EXCLUDE_DATASET_MLAAD_EN}): {len(all_results)}")
    if not all_results:
        print("No results to analyze.")
        return

    # Only keep runs that have metrics for ALL datasets (same as the most complete run)
    run_to_datasets = defaultdict(set)
    for r in all_results:
        run_to_datasets[r["run_dir"]].add(r["dataset"])
    max_count = max(len(s) for s in run_to_datasets.values())
    # Required set = datasets from any run that has the maximum number of datasets (take first such run)
    required_set = None
    for run_dir, ds_set in run_to_datasets.items():
        if len(ds_set) == max_count:
            required_set = ds_set
            break
    if required_set is None:
        required_set = set()
    # Do not require mlaad_en for a run to be "complete" (so runs without mlaad_en still count)
    required_for_complete = required_set - {EXCLUDE_DATASET_MLAAD_EN}
    complete_runs = {run_dir for run_dir, ds_set in run_to_datasets.items() if required_for_complete <= ds_set}
    all_results = [r for r in all_results if r["run_dir"] in complete_runs]
    dropped = len(run_to_datasets) - len(complete_runs)
    if dropped > 0:
        print(f"Excluded {dropped} run(s) that do not have all required datasets (required_set minus mlaad_en). Kept {len(complete_runs)} complete run(s).")
    if not all_results:
        print("No results left after requiring complete runs.")
        return

    # Recompute run_pooled from filtered all_results (only complete runs), excluding mlaad_en for main outputs
    results_main = _filter_out_mlaad_en(all_results)
    run_pooled = {}
    for r in results_main:
        key = (r["llm_label"], r["audio_encoder_label"], r["seed"])
        if key not in run_pooled:
            run_pooled[key] = []
        run_pooled[key].append(r)
    _run_pooled = {}
    for key, rows in run_pooled.items():
        metrics_list = [(row["dataset"], {"accuracy": row["accuracy"], "f1_score": row["f1_score"], "eer": row["eer"], "num_samples": row.get("num_samples")}, "") for row in rows]
        _run_pooled[key] = compute_pooled_weighted_from_metrics_list(metrics_list)
    run_pooled = _run_pooled

    # 1) With mlaad_all (all datasets except mlaad_en)
    print("\n" + "=" * 80)
    print("Output 1: WITH mlaad_all (analyze_final_*.csv, no mlaad_en)")
    print("=" * 80)
    run_analysis(results_main, run_pooled, out_dir, "analyze_final")

    # 2) Without mlaad_all and without mlaad_en
    results_womlaad = [r for r in all_results if r["dataset"] != EXCLUDE_DATASET_MLAAD_ALL and r["dataset"] != EXCLUDE_DATASET_MLAAD_EN]
    # Recompute run_pooled for womlaad: only runs that have at least one non-mlaad_all, non-mlaad_en dataset
    run_pooled_womlaad = {}
    for (llm, enc, seed), pooled in run_pooled.items():
        rows_for_seed = [r for r in all_results if r["llm_label"] == llm and r["audio_encoder_label"] == enc and r["seed"] == seed and r["dataset"] != EXCLUDE_DATASET_MLAAD_ALL and r["dataset"] != EXCLUDE_DATASET_MLAAD_EN]
        if not rows_for_seed:
            continue
        metrics_list = [(r["dataset"], {"accuracy": r["accuracy"], "f1_score": r["f1_score"], "eer": r["eer"], "num_samples": r.get("num_samples")}, "") for r in rows_for_seed]
        run_pooled_womlaad[(llm, enc, seed)] = compute_pooled_weighted_from_metrics_list(metrics_list)
    print("\n" + "=" * 80)
    print("Output 2: WITHOUT mlaad_all (analyze_final_womlaad_*.csv)")
    print("=" * 80)
    run_analysis(results_womlaad, run_pooled_womlaad, out_dir, "analyze_final_womlaad")

    # 3) Only LA19, ITW, LA21 (with pooled)
    results_la19_itw_la21 = [r for r in all_results if r["dataset"] in DATASETS_LA19_ITW_LA21]
    if results_la19_itw_la21:
        run_pooled_la19_itw_la21 = {}
        for (llm, enc, seed), _ in run_pooled.items():
            rows_for_seed = [r for r in results_la19_itw_la21 if r["llm_label"] == llm and r["audio_encoder_label"] == enc and r["seed"] == seed]
            if not rows_for_seed:
                continue
            metrics_list = [(r["dataset"], {"accuracy": r["accuracy"], "f1_score": r["f1_score"], "eer": r["eer"], "num_samples": r.get("num_samples")}, "") for r in rows_for_seed]
            run_pooled_la19_itw_la21[(llm, enc, seed)] = compute_pooled_weighted_from_metrics_list(metrics_list)
        print("\n" + "=" * 80)
        print("Output 3: ONLY la19, itw, la21 (analyze_final_la19_itw_la21_*.csv, with pooled)")
        print("=" * 80)
        run_analysis(results_la19_itw_la21, run_pooled_la19_itw_la21, out_dir, "analyze_final_la19_itw_la21")
    else:
        print("\nNo results for la19/itw/la21 only; skipping analyze_final_la19_itw_la21_*.csv")

    # 4) LA19, ITW, LA21 + mlaad_all (with pooled)
    results_la19_itw_la21_mlaad = [r for r in all_results if r["dataset"] in DATASETS_LA19_ITW_LA21_MLAAD]
    if results_la19_itw_la21_mlaad:
        run_pooled_la19_itw_la21_mlaad = {}
        for (llm, enc, seed), _ in run_pooled.items():
            rows_for_seed = [r for r in results_la19_itw_la21_mlaad if r["llm_label"] == llm and r["audio_encoder_label"] == enc and r["seed"] == seed]
            if not rows_for_seed:
                continue
            metrics_list = [(r["dataset"], {"accuracy": r["accuracy"], "f1_score": r["f1_score"], "eer": r["eer"], "num_samples": r.get("num_samples")}, "") for r in rows_for_seed]
            run_pooled_la19_itw_la21_mlaad[(llm, enc, seed)] = compute_pooled_weighted_from_metrics_list(metrics_list)
        print("\n" + "=" * 80)
        print("Output 4: la19, itw, la21 + mlaad_all (analyze_final_la19_itw_la21_mlaad_*.csv, with pooled)")
        print("=" * 80)
        run_analysis(results_la19_itw_la21_mlaad, run_pooled_la19_itw_la21_mlaad, out_dir, "analyze_final_la19_itw_la21_mlaad")
    else:
        print("\nNo results for la19/itw/la21+mlaad_all; skipping analyze_final_la19_itw_la21_mlaad_*.csv")

    # 5) LA19, ITW, LA21 + mlaad_en (with pooled)
    results_la19_itw_la21_mlaad_en = [r for r in all_results if r["dataset"] in DATASETS_LA19_ITW_LA21_MLAAD_EN]
    if results_la19_itw_la21_mlaad_en:
        run_pooled_la19_itw_la21_mlaad_en = {}
        for r in results_la19_itw_la21_mlaad_en:
            key = (r["llm_label"], r["audio_encoder_label"], r["seed"])
            if key in run_pooled_la19_itw_la21_mlaad_en:
                continue
            rows_for_seed = [x for x in results_la19_itw_la21_mlaad_en if x["llm_label"] == r["llm_label"] and x["audio_encoder_label"] == r["audio_encoder_label"] and x["seed"] == r["seed"]]
            metrics_list = [(x["dataset"], {"accuracy": x["accuracy"], "f1_score": x["f1_score"], "eer": x["eer"], "num_samples": x.get("num_samples")}, "") for x in rows_for_seed]
            run_pooled_la19_itw_la21_mlaad_en[key] = compute_pooled_weighted_from_metrics_list(metrics_list)
        print("\n" + "=" * 80)
        print("Output 5: la19, itw, la21 + mlaad_en (analyze_final_la19_itw_la21_mlaad_en_*.csv, with pooled)")
        print("=" * 80)
        run_analysis(results_la19_itw_la21_mlaad_en, run_pooled_la19_itw_la21_mlaad_en, out_dir, "analyze_final_la19_itw_la21_mlaad_en")
    else:
        print("\nNo results for la19/itw/la21+mlaad_en; skipping analyze_final_la19_itw_la21_mlaad_en_*.csv")

    print("\n" + "=" * 80)
    print("Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
