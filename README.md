# AudioLLM Deepfake Detection

Audio-language-model pipeline for **deepfake / spoofing detection**: fine-tune Qwen2.5 instruct models with Whisper or Wav2Vec2 encoders (LoRA or full fine-tune), optionally inject **OpenSmile** acoustic features, and evaluate on ASVspoof, In-The-Wild, and MLAAD.

**Results hub (HF):** [01Yassine/AudioLLM-Deepfake-Detection](https://huggingface.co/models/01Yassine/AudioLLM-Deepfake-Detection)

---

## Best models

| Model | Avg Macro F1 | Path |
|-------|--------------|------|
| **Baseline (no OpenSmile)** | 91.09% | `outputs/suite_best_no_opensmile/` → Whisper + Qwen-0.5B NoLoRA |
| **OpenSmile after (best overall)** | **94.42%** | `outputs/opensmile_after_best/` → LoRA α=256, Whisper + Qwen-0.5B |

Curated metrics, eval CSVs, and all experiment families: see [`results/`](results/) and [`results/all_results_table.json`](results/all_results_table.json).

---

## Repository layout

```
AudioLLM-Deepfake-Detection/
├── configs/              # YAML training configs (suite, opensmile, …)
├── models/               # AudioDeepfakeLLM (audiollm_base_opensmile.py + Whisper/Wav2Vec/EAT encoders)
├── dataset.py            # Benchmark dataset loader
├── train_tokens.py       # Baseline training (Suite)
├── train_tokens_opensmile.py       # OpenSmile stage 1 (before)
├── train_tokens_opensmile_stage2.py # OpenSmile stage 2 (after)
├── scripts/
│   ├── eval_bf16.py              # Evaluation (accuracy, F1, EER)
│   ├── build_all_results_table.py # Rebuild results/all_results_table.*
│   └── push_results_to_hf.py     # Upload results/ to Hugging Face
├── outputs/              # Best checkpoints only (see below)
└── results/              # Best runs + aggregated tables (~148 GB)
```

### `outputs/` (after cleanup)

Only two checkpoints are kept in-repo:

- **`suite_best_no_opensmile/`** — symlink to best Suite NoLoRA run on `/ds-slt/audio/yelkheir/suite_prompt/…`
- **`opensmile_after_best/`** — OpenSmile-after LoRA-256 Whisper 0.5B

All other raw training outputs (~3 TB) are archived under:

`/ds-slt/audio/yelkheir/AudioLLM-Deepfake-Detection-outputs/`

Legacy / one-off scripts:

`/ds-slt/audio/yelkheir/AudioLLM-Deepfake-Detection-legacy/scripts/`  
(including one-off results builders and maintenance tools under `…/scripts/maintenance/`)

---

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...   # optional; needed for gated models
```

For Qwen-Omni / ms-swift experiments, use `requirements.swift.txt` instead.

---

## Training

Configs live under `configs/` — two reference YAMLs for the best models:

| Config | Training script | Model |
|--------|-----------------|-------|
| `configs/suite_baseline_no_opensmile.yaml` | `train_tokens.py` | Whisper 0.5B NoLoRA (no OpenSmile) |
| `configs/opensmile_after_best.yaml` | `train_tokens_opensmile_stage2.py` | OpenSmile-after LoRA-256 Whisper 0.5B |

All other experiment configs (~430 YAMLs) are archived under  
`/ds-slt/audio/yelkheir/AudioLLM-Deepfake-Detection-legacy/configs/`.

**Baseline (Whisper/Wav2Vec + Qwen + LoRA):**

```bash
python train_tokens.py --config configs/suite_baseline_no_opensmile.yaml
```

**OpenSmile stage 1 (inject features before LLM):**

```bash
python train_tokens_opensmile.py --config configs/configs_opensmile/your_run.yaml
```

**OpenSmile stage 2 (after encoder, best performance):**

```bash
python train_tokens_opensmile_stage2.py \
  --config configs/opensmile_after_best.yaml \
  --pretrained_checkpoint <stage1_run_dir> \
  --checkpoint_epoch best
```

Training writes to `output_dir` in the YAML config. Checkpoints: `lora_best/`, `proj_best.pt`, `audio_encoder_best.pt` (LoRA) or `checkpoint_best.pt` (NoLoRA).

---

## Evaluation

```bash
python scripts/eval_bf16.py \
  --checkpoint_dir outputs/opensmile_after_best \
  --input_jsonl /path/to/test.jsonl \
  --epoch best
```

Produces `{dataset}_best_42.csv` and metrics JSON with **accuracy**, **F1**, and **EER**.

---

## Results curation

Rebuild aggregated tables:

```bash
python scripts/build_all_results_table.py
# → results/all_results_table.json, .csv, .md
```

Upload to Hugging Face:

```bash
export HF_TOKEN=hf_...
python scripts/push_results_to_hf.py
```

---

## Datasets

Standard eval splits (JSONL): `asv19_test`, `itw`, `la21`, `mlaad_en`.  
Macro F1 = unweighted average of Fake-class and Real-class F1.

---

## Citation

If you use these checkpoints or results, cite the project repository and the [HF results hub](https://huggingface.co/models/01Yassine/AudioLLM-Deepfake-Detection).
