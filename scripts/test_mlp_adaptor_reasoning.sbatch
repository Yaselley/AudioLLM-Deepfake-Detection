#!/bin/bash
#SBATCH --job-name=test_mlp_reasoning
#SBATCH --partition=H100,H100-SLT,H100-PCI,B200,H200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/test_mlp_reasoning_%j.out
#SBATCH --error=logs/test_mlp_reasoning_%j.err
#SBATCH --container-image=/enroot/dfki_berlin_speech_shared/universal_df.sqsh
#SBATCH --container-mounts=/opt:/opt,/ds:/ds:ro,/ds-slt:/ds-slt,/netscratch/fkallel:/netscratch/fkallel,/netscratch/fkallel/hf_cache:/root/.cache/huggingface
#SBATCH --container-workdir=/netscratch/fkallel/AudioLLM-Deepfake-Detection
#SBATCH --export=ALL

set -euo pipefail

cd /netscratch/fkallel/AudioLLM-Deepfake-Detection
mkdir -p logs

CONFIG=${1:-reasoning_configs/mlp_adaptor_qwen3.5_0.8B_no_lora_2_hir.yaml}
CHECKPOINT=${2:-output_v25/qwen3.5_0.8B_lora_2_myhir_reasoning/mlp_adaptor_reasoning}
OUTPUT_DIR=${CHECKPOINT%/}/evals

echo "Using config: ${CONFIG}"
echo "Using checkpoint: ${CHECKPOINT}"
echo "Writing outputs to: ${OUTPUT_DIR}"

pip install -r requirements.txt

python3 test_tokens.py \
  --config "${CONFIG}" \
  --checkpoint_dir "${CHECKPOINT}" \
  --epoch best \
  --split test \
  --data_path data/hir_sdd_reasoning_test_clean.json \
  --output_dir "${OUTPUT_DIR}/hir_sdd_reasoning" \
  --resume

python3 test_tokens.py \
  --config "${CONFIG}" \
  --checkpoint_dir "${CHECKPOINT}" \
  --epoch best \
  --split test \
  --data_path data/itw_r.json \
  --output_dir "${OUTPUT_DIR}/itw" \
  --resume

python3 test_tokens.py \
  --config "${CONFIG}" \
  --checkpoint_dir "${CHECKPOINT}" \
  --epoch best \
  --split test \
  --data_path data/asvspoof19_eval_r.json \
  --output_dir "${OUTPUT_DIR}/asvspoof19_eval" \
  --resume

python3 test_tokens.py \
  --config "${CONFIG}" \
  --checkpoint_dir "${CHECKPOINT}" \
  --epoch best \
  --split test \
  --data_path data/mlaad_en_r.json \
  --output_dir "${OUTPUT_DIR}/mlaad_en" \
  --resume
