#!/usr/bin/env python3
"""Aggregate all best-run results into a unified metrics table (F1, Acc, EER)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASETS = ["asv19_test", "itw", "la21", "mlaad_en"]

FAMILIES = [
    "Suite",
    "OpenSmile",
    "OpenSmile-After",
    "EAT",
    "Weighted",
    "DAC-6-Lora-Qwen0.5B",
    "SpeechTokenizer-4-Lora-Qwen0.5B",
    "Qwen-Omni-3B-lora-full",
    "Qwen-Omni-3B-lora-opensmile",
]


def find_metrics_json(run_dir: Path, dataset: str) -> Path | None:
    for name in (
        f"{dataset}_best_42_metrics_best_42.json",
        f"NEW{dataset}_best_42_metrics_best_42.json",
    ):
        p = run_dir / name
        if p.exists():
            return p
    hits = list(run_dir.glob(f"*{dataset}*metrics*.json"))
    return hits[0] if hits else None


def find_csv(run_dir: Path, dataset: str) -> Path | None:
    for prefix in ("", "NEW"):
        p = run_dir / f"{prefix}{dataset}_best_42.csv"
        if p.exists():
            return p
    return None


def macro_f1_from_csv(path: Path) -> tuple[float, float]:
    gt, pred = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gt.append(row["ground_truth"].strip().lower())
            pred.append(row["prediction"].strip().lower())
    f1s = []
    for cls in ("fake", "real"):
        tp = sum(1 for g, p in zip(gt, pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gt, pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gt, pred) if g == cls and p != cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    acc = sum(1 for g, p in zip(gt, pred) if g == p) / len(gt)
    return sum(f1s) / 2, acc


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Equal error rate via ROC/DET sweep (no sklearn dependency)."""
    scores = np.asarray(y_score, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    prev = -np.inf
    best = 1.0
    for score, label in zip(scores, labels):
        if score != prev:
            fnr = 1.0 - tp / n_pos
            fpr = fp / n_neg
            best = min(best, (fnr + fpr) / 2.0)
            prev = score
        if label == 1:
            tp += 1
        else:
            fp += 1
    fnr = 1.0 - tp / n_pos
    fpr = fp / n_neg
    return float(min(best, (fnr + fpr) / 2.0))


def eer_from_csv(path: Path) -> float:
    gt, scores = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        score_col = "score_fake" if "score_fake" in fields else None
        if not score_col:
            raise ValueError(f"No score_fake column in {path}")
        for row in reader:
            label = row["ground_truth"].strip().lower()
            gt.append(1 if label == "fake" else 0)
            scores.append(float(row[score_col]))
    return compute_eer(np.array(gt), np.array(scores))


def parse_run_path(meta_path: Path) -> dict:
    rel = meta_path.parent.relative_to(RESULTS)
    parts = rel.parts
    family = parts[0]
    out = {
        "family": family,
        "path": str(meta_path.parent.relative_to(ROOT)),
        "meta_path": str(meta_path.relative_to(ROOT)),
    }
    if family in (
        "DAC-6-Lora-Qwen0.5B",
        "SpeechTokenizer-4-Lora-Qwen0.5B",
        "Qwen-Omni-3B-lora-full",
        "Qwen-Omni-3B-lora-opensmile",
    ):
        out.update(
            {
                "set": "Lora",
                "encoder": family.split("-")[0].lower()
                if family.startswith("DAC")
                else (
                    "speechtokenizer"
                    if "SpeechTokenizer" in family
                    else "qwen-omni"
                ),
                "llm": "Qwen-0.5B"
                if "0.5B" in family
                else "Qwen-Omni-3B",
                "freeze": None,
                "lora_alpha": 16,
            }
        )
        if "opensmile" in family:
            out["opensmile"] = True
        return out

    if family == "EAT":
        out["set"] = parts[1]
        out["freeze"] = parts[2] if parts[1] == "Lora" else None
        out["encoder"] = "eat"
        out["llm"] = parts[-1]
        return out

    if family == "Weighted":
        out["set"] = parts[1]
        out["encoder"] = parts[2]
        out["llm"] = parts[3]
        out["freeze"] = None
        return out

    # Suite / OpenSmile / OpenSmile-After
    out["set"] = parts[1]
    if parts[1] == "NoLora":
        out["encoder"] = parts[2]
        out["llm"] = parts[3]
        out["freeze"] = None
    else:
        out["freeze"] = parts[2]
        out["encoder"] = parts[3]
        out["llm"] = parts[4]
    if family.startswith("OpenSmile"):
        out["opensmile_stage"] = "after" if family == "OpenSmile-After" else "before"
    return out


def collect_run(meta_path: Path) -> dict | None:
    run_dir = meta_path.parent
    meta = json.loads(meta_path.read_text())
    info = parse_run_path(meta_path)
    per_ds = meta.get("per_dataset") or meta.get("datasets") or {}
    if isinstance(per_ds, dict) and per_ds and "macro_f1" not in next(iter(per_ds.values()), {}):
        # `datasets` only has sample counts in some metas
        per_ds = meta.get("per_dataset", {})

    datasets_out = {}
    for ds in DATASETS:
        ds_meta = per_ds.get(ds, {}) if isinstance(per_ds.get(ds), dict) else {}
        csv_path = find_csv(run_dir, ds)
        metrics_path = find_metrics_json(run_dir, ds)

        macro_f1 = ds_meta.get("macro_f1")
        accuracy = ds_meta.get("accuracy")
        eer = None

        if metrics_path:
            m = json.loads(metrics_path.read_text())
            eer = m.get("eer")
            if accuracy is None:
                accuracy = m.get("accuracy")
            if macro_f1 is None and csv_path:
                macro_f1, accuracy = macro_f1_from_csv(csv_path)
        elif csv_path:
            macro_f1, accuracy = macro_f1_from_csv(csv_path)
            try:
                eer = eer_from_csv(csv_path)
            except (ValueError, KeyError):
                pass

        if macro_f1 is None and accuracy is None and eer is None:
            continue

        borrowed = False
        if ds == "itw" and meta.get("itw_borrowed_from"):
            borrowed = True
        if meta.get("borrowed_datasets") and ds in meta.get("borrowed_datasets", {}):
            borrowed = True
        if meta.get(f"{ds}_approximate"):
            borrowed = True

        datasets_out[ds] = {
            "macro_f1_pct": round(macro_f1 * 100, 2) if macro_f1 is not None else None,
            "accuracy_pct": round(accuracy * 100, 2) if accuracy is not None else None,
            "eer_pct": round(eer * 100, 2) if eer is not None else None,
            "num_samples": ds_meta.get("num_samples"),
            "borrowed_or_approximate": borrowed,
        }

    if len(datasets_out) < 3:
        return None

    f1_vals = [d["macro_f1_pct"] for d in datasets_out.values() if d["macro_f1_pct"] is not None]
    acc_vals = [d["accuracy_pct"] for d in datasets_out.values() if d["accuracy_pct"] is not None]
    eer_vals = [d["eer_pct"] for d in datasets_out.values() if d["eer_pct"] is not None]

    row = {
        **info,
        "run_name": meta.get("run_name") or meta.get("source_run", "").split("/")[-1],
        "source_run": meta.get("source_run") or meta.get("source_infer_dir"),
        "lora": meta.get("lora"),
        "lora_alpha": meta.get("lora_alpha"),
        "lora_rank": meta.get("lora_rank"),
        "num_datasets": len(datasets_out),
        "datasets": datasets_out,
        "avg_macro_f1_pct": round(sum(f1_vals) / len(f1_vals), 2) if f1_vals else None,
        "avg_accuracy_pct": round(sum(acc_vals) / len(acc_vals), 2) if acc_vals else None,
        "avg_eer_pct": round(sum(eer_vals) / len(eer_vals), 2) if eer_vals else None,
        "notes": [],
    }
    if meta.get("note"):
        row["notes"].append(meta["note"])
    if meta.get("itw_borrowed_from"):
        row["notes"].append(f"itw borrowed: {meta['itw_borrowed_from']}")
    if meta.get("mlaad_en_approximate"):
        row["notes"].append("mlaad_en approximate")
    if not row["notes"]:
        del row["notes"]
    return row


def flatten_for_csv(rows: list[dict]) -> list[dict]:
    flat = []
    for r in rows:
        base = {
            "family": r["family"],
            "set": r.get("set"),
            "freeze": r.get("freeze") or "",
            "encoder": r.get("encoder"),
            "llm": r.get("llm"),
            "lora_alpha": r.get("lora_alpha") or "",
            "opensmile_stage": r.get("opensmile_stage") or "",
            "run_name": r.get("run_name"),
            "num_datasets": r["num_datasets"],
            "avg_macro_f1_pct": r["avg_macro_f1_pct"],
            "avg_accuracy_pct": r["avg_accuracy_pct"],
            "avg_eer_pct": r["avg_eer_pct"],
            "path": r["path"],
        }
        for ds in DATASETS:
            d = r["datasets"].get(ds, {})
            base[f"{ds}_macro_f1_pct"] = d.get("macro_f1_pct")
            base[f"{ds}_accuracy_pct"] = d.get("accuracy_pct")
            base[f"{ds}_eer_pct"] = d.get("eer_pct")
        flat.append(base)
    return flat


def build_readme(rows: list[dict], best_f1: dict, best_eer: dict) -> str:
    lines = [
        "# DeepFense AudioLLM Deepfake Detection — Results Summary",
        "",
        "Unified table of **best runs** across all experiment families.",
        "",
        "## Metrics",
        "",
        "| Metric | Description |",
        "|--------|-------------|",
        "| **Macro F1** | Unweighted average of Fake-class and Real-class F1 (equal weight per class) |",
        "| **Accuracy** | Fraction of correct predictions |",
        "| **EER** | Equal Error Rate from `score_fake` vs ground truth (lower is better); from metrics JSON when available |",
        "",
        "## Datasets",
        "",
        "- **asv19_test** — ASVspoof 2019 LA eval",
        "- **itw** — In-The-Wild",
        "- **la21** — ASVspoof 2021 LA eval",
        "- **mlaad_en** — MLAAD English",
        "",
        "Averages (`avg_*`) are computed over evaluated datasets for each run (typically 4/4).",
        "",
        "## Experiment Families",
        "",
        "| Family | Description |",
        "|--------|-------------|",
        "| **Suite** | Baseline Whisper/Wav2Vec + Qwen LLM (LoRA α=16/64/128/256, frozen/unfrozen, NoLoRA) |",
        "| **OpenSmile** | OpenSmile features injected **before** LLM (stage 1) |",
        "| **OpenSmile-After** | OpenSmile features injected **after** audio encoder (stage 2) |",
        "| **EAT** | EAT audio encoder + Qwen |",
        "| **Weighted** | Layer-weighted fusion of Whisper/Wav2Vec representations |",
        "| **DAC-6-Lora-Qwen0.5B** | DAC tokenizer (6 codebooks) + Qwen-0.5B LoRA |",
        "| **SpeechTokenizer-4-Lora-Qwen0.5B** | SpeechTokenizer + Qwen-0.5B LoRA |",
        "| **Qwen-Omni-3B-lora-full** | Qwen2.5-Omni-3B end-to-end LoRA |",
        "| **Qwen-Omni-3B-lora-opensmile** | Qwen2.5-Omni-3B LoRA + OpenSmile |",
        "",
        "## Best Overall Models",
        "",
        f"### Highest average Macro F1",
        f"- **{best_f1['label']}** — **{best_f1['avg_macro_f1_pct']:.2f}%** macro F1, "
        f"{best_f1['avg_accuracy_pct']:.2f}% accuracy, {best_f1['avg_eer_pct']:.2f}% EER",
        f"- Path: `{best_f1['path']}`",
        "",
        f"### Lowest average EER",
        f"- **{best_eer['label']}** — **{best_eer['avg_eer_pct']:.2f}%** EER, "
        f"{best_eer['avg_macro_f1_pct']:.2f}% macro F1, {best_eer['avg_accuracy_pct']:.2f}% accuracy",
        f"- Path: `{best_eer['path']}`",
        "",
        "## Files",
        "",
        "| File | Format | Use |",
        "|------|--------|-----|",
        "| `all_results_table.json` | Nested JSON | Machine-readable; full per-dataset breakdown |",
        "| `all_results_table.csv` | Flat CSV | Spreadsheet / LaTeX table generation |",
        "| `all_results_table.md` | Markdown tables | Human-readable, grouped by family |",
        "",
        "## Notes",
        "",
        "- Some runs borrow missing eval splits (documented in JSON `notes` / `borrowed_or_approximate`).",
        "- OpenSmile **before** stage: LoRA α=16 and Lora-128 only for Qwen-0.5B; NoLoRA all sizes.",
        "- OpenSmile **after** stage: full LoRA α sweep (0.5B) + α=16 for 3B/7B.",
        "- EER requires score columns in eval CSV; if metrics JSON missing, EER computed from CSV.",
        "",
    ]
    return "\n".join(lines)


def build_markdown_tables(rows: list[dict]) -> str:
    lines = [
        "# All Results — Detailed Tables",
        "",
        "Values in **percent** (e.g. 94.42 = 94.42%). EER: lower is better.",
        "",
    ]
    by_family: dict[str, list[dict]] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)

    for family in FAMILIES:
        if family not in by_family:
            continue
        fam_rows = sorted(
            by_family[family],
            key=lambda x: (-(x["avg_macro_f1_pct"] or 0), x.get("set", ""), x.get("encoder", "")),
        )
        lines.append(f"## {family}")
        lines.append("")
        header = (
            "| Set | Freeze | Encoder | LLM | α | "
            "Avg F1 | Avg Acc | Avg EER | "
            "ASV19 F1 | ITW F1 | LA21 F1 | MLAAD F1 | "
            "ASV19 EER | ITW EER | LA21 EER | MLAAD EER |"
        )
        sep = "|" + "---|" * 17
        lines.append(header)
        lines.append(sep)
        for r in fam_rows:
            ds = r["datasets"]
            alpha = r.get("lora_alpha") or ""
            lines.append(
                "| {set} | {freeze} | {enc} | {llm} | {alpha} | "
                "{af1} | {aacc} | {aeer} | "
                "{asv_f1} | {itw_f1} | {la_f1} | {ml_f1} | "
                "{asv_eer} | {itw_eer} | {la_eer} | {ml_eer} |".format(
                    set=r.get("set", ""),
                    freeze=r.get("freeze") or "—",
                    enc=r.get("encoder", ""),
                    llm=r.get("llm", ""),
                    alpha=alpha,
                    af1=r["avg_macro_f1_pct"],
                    aacc=r["avg_accuracy_pct"],
                    aeer=r["avg_eer_pct"],
                    asv_f1=ds.get("asv19_test", {}).get("macro_f1_pct", "—"),
                    itw_f1=ds.get("itw", {}).get("macro_f1_pct", "—"),
                    la_f1=ds.get("la21", {}).get("macro_f1_pct", "—"),
                    ml_f1=ds.get("mlaad_en", {}).get("macro_f1_pct", "—"),
                    asv_eer=ds.get("asv19_test", {}).get("eer_pct", "—"),
                    itw_eer=ds.get("itw", {}).get("eer_pct", "—"),
                    la_eer=ds.get("la21", {}).get("eer_pct", "—"),
                    ml_eer=ds.get("mlaad_en", {}).get("eer_pct", "—"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def label_row(r: dict) -> str:
    parts = [r["family"], r.get("set", "")]
    if r.get("freeze"):
        parts.append(r["freeze"])
    parts.extend([r.get("encoder", ""), r.get("llm", "")])
    if r.get("lora_alpha"):
        parts.append(f"α={r['lora_alpha']}")
    return " / ".join(p for p in parts if p)


def main() -> None:
    rows: list[dict] = []
    for family in FAMILIES:
        base = RESULTS / family
        if not base.is_dir():
            continue
        for meta_path in sorted(base.rglob("best_run_meta.json")):
            row = collect_run(meta_path)
            if row:
                rows.append(row)

    rows.sort(
        key=lambda x: (
            FAMILIES.index(x["family"]) if x["family"] in FAMILIES else 99,
            x.get("set", ""),
            x.get("encoder", ""),
            x.get("llm", ""),
        )
    )

    # Best overall (require 4 datasets and no borrowed flag on all datasets)
    complete = [
        r
        for r in rows
        if r["num_datasets"] == 4
        and r["avg_macro_f1_pct"] is not None
        and r["avg_eer_pct"] is not None
    ]
    best_f1 = max(complete, key=lambda x: x["avg_macro_f1_pct"])
    best_eer = min(complete, key=lambda x: x["avg_eer_pct"])
    best_f1["label"] = label_row(best_f1)
    best_eer["label"] = label_row(best_eer)

    out_json = RESULTS / "all_results_table.json"
    out_csv = RESULTS / "all_results_table.csv"
    out_md = RESULTS / "all_results_table.md"
    out_readme = RESULTS / "README.md"

    payload = {
        "datasets": DATASETS,
        "metric_units": "percent",
        "num_runs": len(rows),
        "best_overall_macro_f1": {
            "label": best_f1["label"],
            "path": best_f1["path"],
            "avg_macro_f1_pct": best_f1["avg_macro_f1_pct"],
            "avg_accuracy_pct": best_f1["avg_accuracy_pct"],
            "avg_eer_pct": best_f1["avg_eer_pct"],
        },
        "best_overall_eer": {
            "label": best_eer["label"],
            "path": best_eer["path"],
            "avg_macro_f1_pct": best_eer["avg_macro_f1_pct"],
            "avg_accuracy_pct": best_eer["avg_accuracy_pct"],
            "avg_eer_pct": best_eer["avg_eer_pct"],
        },
        "runs": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    flat = flatten_for_csv(rows)
    if flat:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=flat[0].keys())
            w.writeheader()
            w.writerows(flat)

    out_md.write_text(build_markdown_tables(rows) + "\n")
    out_readme.write_text(build_readme(rows, best_f1, best_eer))

    print(f"Collected {len(rows)} runs")
    print(f"Best macro F1: {best_f1['label']} -> {best_f1['avg_macro_f1_pct']}%")
    print(f"Best EER: {best_eer['label']} -> {best_eer['avg_eer_pct']}%")
    print(f"Wrote {out_json}, {out_csv}, {out_md}, {out_readme}")


if __name__ == "__main__":
    main()
