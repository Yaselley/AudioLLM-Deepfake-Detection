#!/bin/bash
set -euo pipefail

ROOT=/netscratch/fkallel/AudioLLM-Deepfake-Detection
SCRIPT="$ROOT/scripts/evaluate_reasoning_judge.py"

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

# Was: `while IFS= read -r -d '' eval_dir; do ... done` with no input source —
# that loop had nothing feeding it (mapfile already consumed the find/sort
# pipe above) so it fell back to reading the script's own stdin and just
# blocked forever. Iterate over the array mapfile already built instead.
for eval_dir in "${EVAL_DIRS[@]}"; do
  input="$eval_dir/eval_test_epoch_best.json"
  prompt="$eval_dir/judge_prompts.json"
  scores="$eval_dir/judge_scores.jsonl"

  if [[ ! -f "$input" ]]; then
    echo "Skipping $eval_dir: missing eval_test_epoch_best.json" >&2
    continue
  fi
  echo "Preparing: $input"
  python "$SCRIPT" "$input" \
    --output "$scores" \
    --prompt-file "$prompt" \
    --prepare-only
done