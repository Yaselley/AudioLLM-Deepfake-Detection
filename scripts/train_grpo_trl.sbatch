#!/bin/bash
#SBATCH --job-name=audillm_train
#SBATCH --partition=H100,H100-PCI,H100-SLT,H200,H200-PCI,B200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=logs/audiollm_train_%j.out
#SBATCH --error=logs/audiollm_train_%j.err
#SBATCH --container-image=/enroot/dfki_berlin_speech_shared/universal_df.sqsh
#SBATCH --container-mounts=/opt:/opt,/ds:/ds:ro,/ds-slt:/ds-slt,/netscratch/fkallel:/netscratch/fkallel,/netscratch/fkallel/hf_cache:/root/.cache/huggingface
#SBATCH --container-workdir=/netscratch/fkallel/AudioLLM-Deepfake-Detection
#SBATCH --export=ALL

set -euo pipefail

cd /netscratch/fkallel/AudioLLM-Deepfake-Detection
mkdir -p logs

CONFIG=${1:-config.yaml}
CHECKPOINT=${2}
echo "Running with config: ${CONFIG}"

pip install -r requirements.txt -r requirements.trl.txt
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}
torchrun --standalone --nproc_per_node="${NUM_GPUS}" train_grpo_trl.py \
	--config "${CONFIG}" --pretrained_checkpoint "${CHECKPOINT}"
