#!/bin/bash
#SBATCH --job-name=hir-sdd-judge
#SBATCH --partition=H100,H100-SLT,H100-PCI,H200,H200-PCI
#SBATCH --gres=gpu:4
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hir_sdd_judge_%j.out
#SBATCH --error=logs/hir_sdd_judge_%j.err
#SBATCH --container-image=/enroot/dfki_berlin_speech_shared/universal_df.sqsh
#SBATCH --container-mounts=/opt:/opt,/ds:/ds:ro,/ds-slt:/ds-slt,/netscratch/fkallel:/netscratch/fkallel,/netscratch/fkallel/hf_cache:/root/.cache/huggingface
#SBATCH --container-workdir=/netscratch/fkallel/AudioLLM-Deepfake-Detection
#SBATCH --export=ALL

set -uo pipefail   # note: no -e here — one failed eval dir must not kill the whole job

ROOT=/netscratch/fkallel/AudioLLM-Deepfake-Detection
SCRIPT="$ROOT/scripts/evaluate_reasoning_judge.py"
CONSISTENCY_SCRIPT="$ROOT/scripts/compare_judge_consistency.py"

# Only score the newly added reference judges.
# GPT-OSS was already completed and is intentionally excluded here.
JUDGE_MODELS=(
  # "google/gemma-4-31B-it"
  # "deepseek-ai/DeepSeek-V4-Flash"
  "openai/gpt-oss-120b"
)

# --- Discover every eval dir across all the locations you listed ---
#
# 1) output_v25 / output_v26: <root>/<model>/evals            (as before)
# 2) output_v5/qwen3.5_0.8B_no_lora_2_myhir_asvspoof19: a PARENT to search
#    under, i.e. <that path>/<subdir>/evals — same pattern as v25/v26
# 3) output_v6/qwen3.5_0.8B_lora_{2,42,240}_asv19/{new_new_reasoning_stage2_output,
#    new_new_explanation_stage2_output}: evals inside each stage-output subdir
# 4) output_grpo_trl: <root>/<model>/evals (same pattern as v25/v26), plus
#    output_grpo_trl/qwen3.5_0.8B_no_lora_2_myhir_asvspoof19, which is ALSO a
#    parent to search under (same one-level-deeper pattern as #2)
#
# Adjust the mindepth/maxdepth for any of these that don't match your actual
# layout — the printed list below will make a mismatch obvious immediately.
mapfile -d '' EVAL_DIRS < <(
  {
    find "$ROOT/output_v25" "$ROOT/output_v26" \
      -mindepth 2 -maxdepth 2 -type d -name evals -print0 2>/dev/null

    find "$ROOT/output_v5/qwen3.5_0.8B_no_lora_2_myhir_asvspoof19" \
      -mindepth 2 -maxdepth 2 -type d -name evals -print0 2>/dev/null

    find "$ROOT"/output_v6/qwen3.5_0.8B_lora_{2,42,240}_asv19/{new_new_reasoning_stage2_output,new_new_explanation_stage2_output} \
      -mindepth 1 -maxdepth 1 -type d -name evals -print0 2>/dev/null

    find "$ROOT/output_grpo_trl" \
      -mindepth 2 -maxdepth 2 -type d -name evals -print0 2>/dev/null

    find "$ROOT/output_grpo_trl/qwen3.5_0.8B_no_lora_2_myhir_asvspoof19" \
      -mindepth 2 -maxdepth 2 -type d -name evals -print0 2>/dev/null
  } | sort -z -u
)

echo "Found ${#EVAL_DIRS[@]} eval directories to process:"
printf '  %s\n' "${EVAL_DIRS[@]}"
echo

if [[ ${#EVAL_DIRS[@]} -eq 0 ]]; then
  echo "No eval directories found — check the paths above before running the judges." >&2
  exit 1
fi

cd "$ROOT"
pip uninstall -y vllm flashinfer triton torch torchvision torchaudio transformers
pip install --no-cache-dir \
  torch==2.11.0 \
  torchaudio==2.11.0 \
  torchvision==0.26.0 \
  transformers==5.14.1 \
  "vllm==0.23.0"
pip install --no-cache-dir nvidia-cuda-nvrtc-cu13

for eval_dir in "${EVAL_DIRS[@]}"; do
  echo "=== Processing $eval_dir ==="
  input="$eval_dir/eval_test_epoch_best.json"
  prompt="$eval_dir/judge_prompts.json"

  if [[ ! -f "$input" ]]; then
    echo "  Missing input: $input — skipping" >&2
    continue
  fi
  if [[ ! -f "$prompt" ]]; then
    echo "  Missing prompts: $prompt; run prepare_reasoning_judge_all.sh first — skipping" >&2
    continue
  fi

  score_files=()
  declare -A seen_score_files=()

  for model in "${JUDGE_MODELS[@]}"; do
    model_slug="${model//\//_}"
    model_slug="${model_slug//:/_}"
    judge_output="$eval_dir/judge_gpt_oss_120b_outputs.jsonl"
    scores="$eval_dir/judge_scores.jsonl"

    if [[ -f "$scores" ]]; then
      if [[ -z "${seen_score_files[$scores]:-}" ]]; then
        score_files+=("$scores")
        seen_score_files["$scores"]=1
      fi
    fi

    if [[ -f "$judge_output" && -s "$judge_output" ]]; then
      echo "  Skipping model $model because output already exists at $judge_output"
    else
      resume=""
      [[ -f "$judge_output" ]] && resume="--resume"

      echo "  Running judge model: $model"
      extra_args=()
      if [[ "$model" == deepseek-ai/DeepSeek-V4-Flash* ]]; then
        extra_args+=(--kv-cache-dtype fp8)
      fi

      python "$ROOT/scripts/run_vllm_reasoning_judge.py" \
        "$prompt" \
        --output "$judge_output" \
        --model "$model" \
        --tensor-parallel-size 4 \
        --max-tokens 2048 \
        --max-model-len 4096 \
        "${extra_args[@]}" \
        $resume
      fi

    if [[ -f "$scores" ]]; then
      echo "  Using existing score file: $scores"
    else
      python "$SCRIPT" "$input" \
        --judge-output "$judge_output" \
        --output "$scores" \
        --prompt-file "$prompt" \
        --model "$model"
    fi
  done

  for existing_score in "$eval_dir"/judge_scores*.jsonl; do
    [[ -e "$existing_score" ]] || continue
    if [[ -z "${seen_score_files[$existing_score]:-}" ]]; then
      score_files+=("$existing_score")
      seen_score_files["$existing_score"]=1
    fi
  done

  # if (( ${#score_files[@]} > 0 )); then
  #   python "$CONSISTENCY_SCRIPT" \
  #     --score-files "${score_files[@]}" \
  #     --output "$eval_dir/judge_consistency.json"
  # else
  #   echo "  No judge score files produced for consistency comparison." >&2
  # fi

  echo
done

echo "All eval directories processed."