#!/bin/bash
#SBATCH --job-name=test_qwen3.5_0.8B_lora_2
#SBATCH --partition=H100,H100-SLT,H100-PCI,B200,H200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/test_qwen3.5_0.8B_lora_2_%j.out
#SBATCH --error=logs/test_qwen3.5_0.8B_lora_2_%j.err
#SBATCH --container-image=/enroot/dfki_berlin_speech_shared/universal_df.sqsh
#SBATCH --container-mounts=/opt:/opt,/ds:/ds:ro,/ds-slt:/ds-slt,/netscratch/fkallel:/netscratch/fkallel,/netscratch/fkallel/hf_cache:/root/.cache/huggingface
#SBATCH --container-workdir=/netscratch/fkallel/AudioLLM-Deepfake-Detection
#SBATCH --export=ALL

set -euo pipefail

cd /netscratch/fkallel/AudioLLM-Deepfake-Detection
mkdir -p logs
CONFIG=${1:-config.yaml}
CHECKPOINT=${2}
OUTPUT_DIR=${CHECKPOINT%/}/evals


pip install -r requirements.txt
python3 test_tokens.py \
  --config "${CONFIG}" \
  --checkpoint_dir "${CHECKPOINT}" \
  --epoch best \
  --split test \
  --data_path data/my_hir_sdd_binary_test_unique_2.json \
  --output_dir "${OUTPUT_DIR}"
python3 test_tokens.py \
  --config "${CONFIG}" \
  --checkpoint_dir  "${CHECKPOINT}" \
  --epoch best \
  --split test \
  --data_path data/itw.json \
  --output_dir "${OUTPUT_DIR}/itw"
python3 test_tokens.py \
  --config "${CONFIG}" \
  --checkpoint_dir "${CHECKPOINT}" \
  --epoch best \
  --split test \
  --data_path data/asvspoof19_eval.json \
  --output_dir "${OUTPUT_DIR}/asvspoof19_eval"\
  --resume

python3 test_tokens.py \
  --config "${CONFIG}" \
  --checkpoint_dir "${CHECKPOINT}" \
  --epoch best \
  --split test \
  --data_path data/mlaad_en.json \
  --output_dir  "${OUTPUT_DIR}/mlaad_en"\
  --resume

