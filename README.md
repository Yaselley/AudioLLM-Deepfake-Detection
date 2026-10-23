# AudioLLM Deepfake Detection

Code and experiment configs for my master's thesis on **speech deepfake detection with
audio-language models**. The project studies how the *supervision format* — a plain
Real/Fake label, a reasoning trace before the verdict, or a verdict followed by a
post-hoc explanation — affects detection and out-of-domain generalisation, and whether
GRPO after supervised fine-tuning (SFT) helps.

The detector is an **XLS-R 300M** speech encoder, a **linear projector**, and a
**Qwen3.5** instruction-tuned LLM (0.8B / 4B). Audio is presented to the LLM as a
continuous prefix; there is no speech tokenizer and no hand-crafted acoustic feature
front-end.

## What the thesis covers

1. **Binary supervision** — SFT on short `Real` / `Fake` targets, varying LLM size
   (0.8B / 4B) and LLM adaptation (frozen vs. LoRA), with the encoder frozen or fully
   fine-tuned.
2. **Reasoning supervision** — SFT on structured rationales
   (`<observations>`, `<reasoning_step>`, `<conclusion>`, `<answer>`) where the
   reasoning comes *before* the verdict.
3. **Explanation supervision** — same schema, but the verdict comes first and the
   rationale is a post-hoc explanation.
4. **GRPO after SFT** — TRL GRPO with a shaped reward combining label correctness,
   output format, and token-level similarity to the reference rationale.
5. **Reasoning-quality evaluation** — three LLM judges score generated rationales on
   correctness, reasoning quality, comprehensiveness, and logic (1–10).

## Architecture

```
raw 16 kHz audio (10 s)
        │
   XLS-R 300M (fairseq wav2vec2 backbone; frozen or fully fine-tuned)
        │  frame-level hidden states
   Linear projector  (encoder dim → LLM hidden size)
        │
   [BOS] + prompt text + projected speech + prompt text
        │
   Qwen3.5 0.8B / 4B  (frozen or LoRA: rank 64, α 16, dropout 0.05)
        │
   generated answer / rationale
```

- Implemented in `models/audiollm.py` (`AudioDeepfakeLLM`) and
  `models/speech_encoders/` (`Wav2Vec2Encoder`, used for XLS-R).
- SFT uses next-token cross-entropy on the completion only; prompt tokens are masked
  with `-100`.
- The optional MLP adaptor (`mlp_adaptor`) is retained as an exploratory head on the
  pooled final hidden state, but is not part of the main thesis narrative.
- The LLM adaptation is either "frozen" (only encoder + projector are trained) or LoRA
  on the attention and MLP projections.

## Data

All data files live under `data/`. Paths inside configs point at the cluster layout
used for the experiments and must be adapted locally.

| Purpose | Files |
| --- | --- |
| HIR-SDD binary | `my_hir_sdd_binary_{train,val,test}_unique_2.json` |
| HIR-SDD reasoning (before answer) | `hir_sdd_reasoning_{train,val,test}_clean.json` |
| HIR-SDD explanation (answer first) | `hir_sdd_explanation_{train,val,test}_clean.json` |
| ASVspoof 2019 LA binary | `asvspoof19_{train,dev,eval}.json` |
| ASVspoof 2019 LA reasoning | `asvspoof19_train_reasoning.json`, `asvspoof19_eval_r.json` |
| In-the-Wild | `itw.json`, `itw_r.json` |
| MLAAD v3 (English) | `mlaad_en.json`, `mlaad_en_r.json` |

HIR-SDD uses **source-disjoint** train/validation/test splits. Reasoning annotations
were restructured with `Qwen3-30B-A3B-Instruct-2507` and filtered for label leakage with
`Qwen3.5-35B-A3B`; ASVspoof 2019 LA rationales were generated with
`Qwen3-Omni-30B-A3B-Instruct`.

Prompts live in `prompts/`:

- `qwen_prompt.json` — binary Real/Fake question.
- `train_prompt.json` — reasoning-before-answer schema.
- `train_prompt_explanation.json` — answer-then-explanation schema.

`dataset.py` loads 10-second clips at 16 kHz, selects the
`deepfake_detection_with_reasoning` prompt when present, and otherwise falls back to the
binary prompt.

## Repository layout

```
models/
  audiollm.py                 # XLS-R + projector + Qwen3.5 (+ LoRA, optional MLP head)
  speech_encoders/
    base.py                   # encoder interface
    wav2vec2_encoder.py       # XLS-R / Wav2Vec2 (fairseq or HuggingFace)
    weighted_layer_pooling.py
dataset.py                    # audio + prompt + target loading
train_common.py               # model building, optimiser, training/validation loops
train_tokens.py               # SFT entry point
train_continue.py             # continued SFT from a pretrained checkpoint
train_grpo_trl.py             # GRPO after SFT (TRL)
trl_grpo.py                   # custom audio GRPO trainer
grpo_utils.py                 # shaped reward (label / format / reasoning)
test_tokens.py                # generation-based evaluation -> JSON + metrics
configs/                      # binary SFT configs (0.8B / 4B, lora / frozen, seeds)
reasoning_configs/            # reasoning-before-answer SFT configs
explanation_configs/          # answer-then-explanation SFT configs
grpo_reasoning_configs/       # GRPO configs
scripts/                      # Slurm jobs, metrics, LLM-judge utilities
data/                         # datasets and annotations
prompts/                      # prompt templates
```

## Setup

```bash
pip install -r requirements.txt
# GRPO additionally needs a TRL-compatible stack:
pip install -r requirements.trl.txt
```

XLS-R is loaded through fairseq (`audio_encoder_source: fairseq`), or through
HuggingFace `Wav2Vec2Model` (`audio_encoder_source: huggingface`). Set `HF_TOKEN` if the
Qwen checkpoints are gated.

## Training

### 1. Supervised fine-tuning

```bash
python train_tokens.py --config configs/qwen3.5_0.8B_lora_42.yaml
```

Configs are grouped by supervision type:

- `configs/` — binary detection (`prompts/qwen_prompt.json`).
- `reasoning_configs/` — reasoning before the verdict.
- `explanation_configs/` — verdict followed by an explanation.

Each family covers Qwen3.5 0.8B and 4B, LoRA and frozen LLM, seeds 2 / 42 / 240, and
HIR-SDD-only, HIR-SDD + ASVspoof 2019, and ASVspoof-only training sets. The default SFT
recipe is 5 epochs, AdamW with weight decay 0.05, encoder LR `1e-6`, projector/LoRA LR
`1e-5`, and bf16.

Checkpoints are written under the config's `output_dir` as `lora_best/`, `proj_best.pt`,
`audio_encoder_best.pt`, plus one directory per epoch under `epoch_ckpts/`.

### 2. Continued SFT from a checkpoint

```bash
python train_continue.py \
  --config configs/qwen3.5_0.8B_lora_42.yaml \
  --pretrained_checkpoint output_v5/qwen3.5_0.8B_lora_42_myhir_asvspoof19 \
  --checkpoint_epoch best \
  --output_dir output_v6/continued
```

This loads the projector, encoder, and LoRA adapter from a previous run, re-enables the
trainable parameter groups, and starts a second SFT stage (for example LA19 →
HIR-SDD). It is also used to continue training for one extra epoch.

### 3. GRPO after SFT

```bash
python train_grpo_trl.py \
  --config grpo_reasoning_configs/qwen3.5_0.8B_lora_42_hir_trl.yaml \
  --pretrained_checkpoint output_v5/qwen3.5_0.8B_lora_42_myhir_asvspoof19
```

The GRPO stage starts from an SFT checkpoint and uses TRL. Typical settings: 4 rollouts,
`max_new_tokens` 150, temperature 0.8, top-p 0.9, repetition penalty 1.1, clip 0.2, KL
coefficient 0.1, one epoch. The reward is `1.0` for a correct label, `0.1` for a valid
output format, and `0.1` for token-F1 similarity between the generated and reference
reasoning (`grpo_utils.py`).

Slurm wrappers are provided in `scripts/` (`train_tokens.sbatch`,
`train_after*.sbatch`, `train_grpo_trl.sbatch`).

## Evaluation

### Label metrics

`test_tokens.py` generates outputs for a split and writes
`eval_<split>_epoch_<epoch>.json` plus `..._metrics.json` (accuracy, macro-F1, and
confusion counts). It supports `--resume` for wall-clock-limited jobs.

```bash
python test_tokens.py \
  --config configs/qwen3.5_0.8B_lora_42.yaml \
  --checkpoint_dir output_v5/qwen3.5_0.8B_lora_42_myhir_asvspoof19 \
  --split test \
  --data_path data/hir_sdd_reasoning_test_clean.json \
  --output_dir output_v5/qwen3.5_0.8B_lora_42_myhir_asvspoof19/evals
```

The four thesis benchmarks are HIR-SDD (source-disjoint test), In-the-Wild, ASVspoof
2019 LA, and MLAAD v3 English. `scripts/test_reasoning.sbatch` runs all four.

To recompute metrics from saved predictions, or to evaluate outputs produced elsewhere:

```bash
python scripts/evaluate_jsonl.py path/to/eval_test_epoch_best.json --output metrics.json
```

`scripts/eval_bf16.py` is an older logit-based binary evaluator (Real/Fake token
probabilities) and is kept as a utility; the generation-based path above is the primary
evaluation used in the thesis.

### Reasoning quality (LLM judges)

```bash
# 1. Prepare judge prompts from saved generations
python scripts/evaluate_reasoning_judge.py \
  output/evals/eval_test_epoch_best.json \
  --output output/evals/judge_scores.jsonl \
  --prompt-file output/evals/judge_prompts.json \
  --prepare-only

# 2. Run a judge (Groq API or local vLLM)
python scripts/run_vllm_reasoning_judge.py \
  output/evals/judge_prompts.json \
  --model openai/gpt-oss-120b \
  --output output/evals/judge_raw.jsonl

# 3. Merge judge scores back into the per-record results
python scripts/evaluate_reasoning_judge.py \
  output/evals/eval_test_epoch_best.json \
  --output output/evals/judge_scores.jsonl \
  --judge-output output/evals/judge_raw.jsonl
```

Judges score **Correctness**, **Reasoning Quality**, **Comprehensiveness**, and
**Logic** on a 1–10 scale. The thesis uses `gpt-oss-120B`, `gemma-4-31B`, and
`Selene-1-Llama-3.3-70B`; `scripts/prepare_reasoning_judge_all.sh` batches preparation over a directory
tree of evaluation runs.

## Selected results

Macro-F1 (%) on the thesis benchmarks, averaged over three seeds for SFT and reported
for single runs for GRPO (GRPO is substantially more expensive). Full tables are in the
thesis.

| Setting | HIR-SDD | ITW | LA19 | MLAAD | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| Binary, 0.8B Frozen, HIR-SDD + LA19 | 94.61 | 69.24 | 97.06 | 99.01 | **89.98** |
| Reasoning SFT, 0.8B LoRA, HIR-SDD | 87.98 | 69.98 | 62.88 | 98.22 | 79.76 |
| Explanation SFT, 0.8B LoRA, HIR-SDD | 90.09 | 46.26 | 50.96 | 97.84 | 71.29 |
| SFT → GRPO, 0.8B LoRA, Reasoning HIR-SDD (seed 42) | 89.94 | 89.29 | 94.49 | 96.51 | **92.56** |
| SFT → GRPO, continued on HIR-SDD + LA19 reasoning (seed 42) | 86.81 | 89.91 | 99.10 | 97.04 | **93.22** |

Main findings:

- Reasoning-before-answer supervision transfers better than answer-then-explanation.
- Neither a larger LLM nor LoRA consistently improves over a frozen 0.8B LLM.
- GRPO after SFT is effective with LoRA and unreliable with a frozen LLM.

## Notes

- All paths in configs and Slurm scripts use the original cluster layout
  (`/netscratch/fkallel`, `/ds-slt`, ...). Adjust `data.*`, `model.llm_path`,
  `model.audio_encoder_path`, and `output_dir` for your environment.
- Utility scripts under `scripts/` (counting predictions, printing samples,
  moving/re-labelling records) are experiment helpers, not part of the core pipeline.
