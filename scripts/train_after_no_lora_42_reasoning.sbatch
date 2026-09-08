#!/bin/bash
#SBATCH --job-name=audillm_train
#SBATCH --partition=H100,H100-SLT,H100-PCI,B200,H200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
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
OUTPUT_DIR=${CHECKPOINT%/}/no_lora_42_reasoning_stage2_output
echo "Running with config: ${CONFIG}"

pip install -r requirements.txt
python3 train_tokens_opensmile_stage2.py --config "${CONFIG}" --pretrained_checkpoint "${CHECKPOINT}" --output_dir "${OUTPUT_DIR}"
