#!/bin/bash
#SBATCH --job-name=audillm_mlp_reasoning
#SBATCH --partition=H100,H100-SLT,H100-PCI,B200,H200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/audiollm_mlp_reasoning_%j.out
#SBATCH --error=logs/audiollm_mlp_reasoning_%j.err
#SBATCH --container-image=/enroot/dfki_berlin_speech_shared/universal_df.sqsh
#SBATCH --container-mounts=/opt:/opt,/ds:/ds:ro,/ds-slt:/ds-slt,/netscratch/fkallel:/netscratch/fkallel,/netscratch/fkallel/hf_cache:/root/.cache/huggingface
#SBATCH --container-workdir=/netscratch/fkallel/AudioLLM-Deepfake-Detection
#SBATCH --export=ALL

set -euo pipefail

cd /netscratch/fkallel/AudioLLM-Deepfake-Detection
mkdir -p logs

CONFIG=${1:?Usage: sbatch $0 CONFIG CHECKPOINT [CHECKPOINT_EPOCH] [OUTPUT_DIR]}
CHECKPOINT=${2:?Usage: sbatch $0 CONFIG CHECKPOINT [CHECKPOINT_EPOCH] [OUTPUT_DIR]}
CHECKPOINT_EPOCH=${3:-best}
OUTPUT_DIR=${4:-${CHECKPOINT%/}/mlp_adaptor_reasoning}

echo "Running MLP adaptor training with config: ${CONFIG}"
echo "Loading reasoning checkpoint: ${CHECKPOINT} (${CHECKPOINT_EPOCH})"
echo "Writing adaptor checkpoint to: ${OUTPUT_DIR}"

pip install -r requirements.txt
python3 train_continue.py \
	--config "${CONFIG}" \
	--pretrained_checkpoint "${CHECKPOINT}" \
	--checkpoint_epoch "${CHECKPOINT_EPOCH}" \
	--output_dir "${OUTPUT_DIR}"