#!/bin/bash
#SBATCH --job-name=test_qwen3.5_0.8B_lora_2
#SBATCH --partition=H100,H100-SLT,H100-PCI,B200,H200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/test_qwen3.5_0.8B_lora_2_%j.out
#SBATCH --error=logs/test_qwen3.5_0.8B_lora_2_%j.err
#SBATCH --container-image=/enroot/dfki_berlin_speech_shared/universal_df.sqsh
#SBATCH --container-mounts=/opt:/opt,/ds:/ds:ro,/ds-slt:/ds-slt,/netscratch/fkallel:/netscratch/fkallel,/netscratch/fkallel/hf_cache:/root/.cache/huggingface
#SBATCH --container-workdir=/netscratch/fkallel/AudioLLM-Deepfake-Detection
#SBATCH --export=ALL

set -euo pipefail

cd /netscratch/fkallel/AudioLLM-Deepfake-Detection
mkdir -p logs output_v6/evals

pip install -r requirements.txt
python3 test_tokens.py \
  --config grpo_reasoning_configs/qwen3.5_0.8B_lora_2_hir.yaml \
  --checkpoint_dir output_grpo/qwen3.5_0.8B_lora_2_myhir_after_asvspoof19/step_ckpt \
  --epoch best \
  --split test \
  --data_path /netscratch/fkallel/SALMONN/data/my_hir_sdd_test_2_structured.json \
  --output_dir output_grpo/qwen3.5_0.8B_lora_2_myhir_after_asvspoof19/step_ckpt/evals_after
python3 test_tokens.py \
  --config grpo_reasoning_configs/qwen3.5_0.8B_lora_2_hir.yaml \
  --checkpoint_dir output_grpo/qwen3.5_0.8B_lora_2_myhir_after_asvspoof19/step_ckpt \
  --epoch best \
  --split test \
  --data_path data/itw_r.json \
  --output_dir output_grpo/qwen3.5_0.8B_lora_2_myhir_after_asvspoof19/step_ckpt/evals_after/itw
python3 test_tokens.py \
  --config grpo_reasoning_configs/qwen3.5_0.8B_lora_2_hir.yaml \
  --checkpoint_dir output_grpo/qwen3.5_0.8B_lora_2_myhir_after_asvspoof19/step_ckpt \
  --epoch best \
  --split test \
  --data_path data/asvspoof19_eval_r.json \
  --output_dir output_grpo/qwen3.5_0.8B_lora_2_myhir_after_asvspoof19/step_ckpt/evals_after/asvspoof19_eval\
  --resume

python3 test_tokens.py \
  --config grpo_reasoning_configs/qwen3.5_0.8B_lora_2_hir.yaml \
  --checkpoint_dir output_grpo/qwen3.5_0.8B_lora_2_myhir_after_asvspoof19/step_ckpt \
  --epoch best \
  --split test \
  --data_path data/mlaad_en_r.json \
  --output_dir  output_grpo/qwen3.5_0.8B_lora_2_myhir_after_asvspoof19/step_ckpt/mlaad_en\
  --resume

