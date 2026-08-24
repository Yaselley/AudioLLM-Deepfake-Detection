#!/bin/bash
#SBATCH --job-name=test_qwen3.5_4B_lora_240
#SBATCH --partition=H100,H100-SLT,H100-PCI,B200,H200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/test_qwen3.5_4B_lora_240_%j.out
#SBATCH --error=logs/test_qwen3.5_4B_lora_240_%j.err
#SBATCH --container-image=/enroot/dfki_berlin_speech_shared/universal_df.sqsh
#SBATCH --container-mounts=/opt:/opt,/ds:/ds:ro,/ds-slt:/ds-slt,/netscratch/fkallel:/netscratch/fkallel,/netscratch/fkallel/hf_cache:/root/.cache/huggingface
#SBATCH --container-workdir=/netscratch/fkallel/AudioLLM-Deepfake-Detection
#SBATCH --export=ALL

set -euo pipefail

cd /netscratch/fkallel/AudioLLM-Deepfake-Detection
mkdir -p logs output_v6/evals

pip install -r requirements.txt

python3 test_tokens.py \
  --config explanation_configs/qwen3.5_4B_lora_240_hir.yaml \
  --checkpoint_dir output_v24/qwen3.5_4B_lora_240_myhir_reasoning/stage2_explanation_output \
  --epoch best \
  --split test \
  --data_path data/mlaad_en_r.json \
  --output_dir output_v24/qwen3.5_4B_lora_240_myhir_reasoning/stage2_explanation_output/evals/mlaad_en\
  --resume


