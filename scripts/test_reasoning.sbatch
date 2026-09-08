#!/bin/bash
#SBATCH --job-name=test_reasoning
#SBATCH --partition=H100,H100-SLT,H100-PCI,B200,H200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/test_reasoning_%j.out
#SBATCH --error=logs/test_reasoning_%j.err
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
  --data_path data/hir_sdd_reasoning_test_clean.json \
  --output_dir "${OUTPUT_DIR}"\
  --resume
python3 test_tokens.py \
  --config "${CONFIG}" \
  --checkpoint_dir  "${CHECKPOINT}" \
  --epoch best \
  --split test \
  --data_path data/itw_r.json \
  --output_dir "${OUTPUT_DIR}/itw"\
  --resume
python3 test_tokens.py \
  --config "${CONFIG}" \
  --checkpoint_dir "${CHECKPOINT}" \
  --epoch best \
  --split test \
  --data_path data/asvspoof19_eval_r.json \
  --output_dir "${OUTPUT_DIR}/asvspoof19_eval"\
  --resume

python3 test_tokens.py \
  --config "${CONFIG}" \
  --checkpoint_dir "${CHECKPOINT}" \
  --epoch best \
  --split test \
  --data_path data/mlaad_en_r.json \
  --output_dir  "${OUTPUT_DIR}/mlaad_en"\
  --resume

