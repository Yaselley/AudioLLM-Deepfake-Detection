#!/bin/bash
#SBATCH --job-name=hir-sdd-judge
#SBATCH --partition=H100,H100-SLT,H100-PCI,H200,H200-PCI
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hir_sdd_judge_%j.out
#SBATCH --error=logs/hir_sdd_judge_%j.err
#SBATCH --container-image=/enroot/dfki_berlin_speech_shared/universal_df.sqsh
#SBATCH --container-mounts=/opt:/opt,/ds:/ds:ro,/ds-slt:/ds-slt,/netscratch/fkallel:/netscratch/fkallel,/netscratch/fkallel/hf_cache:/root/.cache/huggingface
#SBATCH --container-workdir=/netscratch/fkallel/AudioLLM-Deepfake-Detection
#SBATCH --export=ALL

cd /netscratch/fkallel/AudioLLM-Deepfake-Detection
pip install ms-swift qwen-omni-utils qwen_vl_utils decord -U
swift infer \
  --infer_backend transformers \
  --max_batch_size 8 \
  --max_new_tokens 64 \
  --val_dataset output_v25/qwen3.5_0.8B_lora_2_myhir_reasoning/evals/judge_prompts.json \
  --result_path output_v25/qwen3.5_0.8B_lora_2_myhir_reasoning/evals/judge_outputs.jsonl \
  --model Qwen/Qwen3.5-35B-A3B \
  --temperature 0 \
  --write_batch_size 100