#!/bin/bash
#SBATCH --job-name=test_qwen3.5_4B_lora_42
#SBATCH --partition=H100,H100-SLT,H100-PCI,B200,H200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/test_qwen3.5_4B_lora_42_%j.out
#SBATCH --error=logs/test_qwen3.5_4B_lora_42_%j.err
#SBATCH --container-image=/enroot/dfki_berlin_speech_shared/universal_df.sqsh
#SBATCH --container-mounts=/opt:/opt,/ds:/ds:ro,/ds-slt:/ds-slt,/netscratch/fkallel:/netscratch/fkallel,/netscratch/fkallel/hf_cache:/root/.cache/huggingface
#SBATCH --container-workdir=/netscratch/fkallel/AudioLLM-Deepfake-Detection
#SBATCH --export=ALL

set -euo pipefail

cd /netscratch/fkallel/AudioLLM-Deepfake-Detection
mkdir -p logs output_v6/evals

pip install -r requirements.txt
python3 test_tokens.py \
  --config configs/qwen3.5_4B_lora_42_asv19.yaml \
  --checkpoint_dir output_v6/qwen3.5_4B_lora_42_asv19 \
  --epoch best \
  --split test \
  --data_path data/my_hir_sdd_binary_test_unique_2.json \
  --output_dir output_v6/qwen3.5_4B_lora_42_asv19/evals
python3 test_tokens.py \
  --config configs/qwen3.5_4B_lora_42_asv19.yaml \
  --checkpoint_dir output_v6/qwen3.5_4B_lora_42_asv19 \
  --epoch best \
  --split test \
  --data_path data/itw.json \
  --output_dir output_v6/qwen3.5_4B_lora_42_asv19/evals/itw
python3 test_tokens.py \
  --config configs/qwen3.5_4B_lora_42_asv19.yaml \
  --checkpoint_dir output_v6/qwen3.5_4B_lora_42_asv19 \
  --epoch best \
  --split test \
  --data_path data/asvspoof19_eval.json \
  --output_dir output_v6/qwen3.5_4B_lora_42_asv19/evals/asvspoof19_eval

python3 test_tokens.py \
  --config configs/qwen3.5_4B_lora_42_asv19.yaml \
  --checkpoint_dir output_v6/qwen3.5_4B_lora_42_asv19 \
  --epoch best \
  --split test \
  --data_path data/mlaad_en.json \
  --output_dir output_v6/qwen3.5_4B_lora_42_asv19/evals/mlaad_en
