#!/bin/bash

cd /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection
mkdir -p logs_suite

python train.py --config configs_suite/qwen2.5_0.5b_w2v_frozen_seed1234.yaml > logs_suite/qwen2.5_0.5b_w2v_frozen_seed1234.txt 2>&1
python train.py --config configs_suite/qwen2.5_0.5b_w2v_frozen_seed2.yaml > logs_suite/qwen2.5_0.5b_w2v_frozen_seed2.txt 2>&1
python train.py --config configs_suite/qwen2.5_0.5b_w2v_frozen_seed42.yaml > logs_suite/qwen2.5_0.5b_w2v_frozen_seed42.txt 2>&1
python train.py --config configs_suite/qwen2.5_0.5b_w2v_unfrozen_seed1234.yaml > logs_suite/qwen2.5_0.5b_w2v_unfrozen_seed1234.txt 2>&1
python train.py --config configs_suite/qwen2.5_0.5b_w2v_unfrozen_seed2.yaml > logs_suite/qwen2.5_0.5b_w2v_unfrozen_seed2.txt 2>&1
python train.py --config configs_suite/qwen2.5_0.5b_w2v_unfrozen_seed42.yaml > logs_suite/qwen2.5_0.5b_w2v_unfrozen_seed42.txt 2>&1
python train.py --config configs_suite/qwen2.5_0.5b_whisperv3_frozen_seed1234.yaml > logs_suite/qwen2.5_0.5b_whisperv3_frozen_seed1234.txt 2>&1
python train.py --config configs_suite/qwen2.5_0.5b_whisperv3_frozen_seed2.yaml > logs_suite/qwen2.5_0.5b_whisperv3_frozen_seed2.txt 2>&1
python train.py --config configs_suite/qwen2.5_0.5b_whisperv3_frozen_seed42.yaml > logs_suite/qwen2.5_0.5b_whisperv3_frozen_seed42.txt 2>&1
python train.py --config configs_suite/qwen2.5_0.5b_whisperv3_unfrozen_seed1234.yaml > logs_suite/qwen2.5_0.5b_whisperv3_unfrozen_seed1234.txt 2>&1
python train.py --config configs_suite/qwen2.5_0.5b_whisperv3_unfrozen_seed2.yaml > logs_suite/qwen2.5_0.5b_whisperv3_unfrozen_seed2.txt 2>&1
python train.py --config configs_suite/qwen2.5_0.5b_whisperv3_unfrozen_seed42.yaml > logs_suite/qwen2.5_0.5b_whisperv3_unfrozen_seed42.txt 2>&1
python train.py --config configs_suite/qwen2.5_3b_w2v_frozen_seed1234.yaml > logs_suite/qwen2.5_3b_w2v_frozen_seed1234.txt 2>&1
python train.py --config configs_suite/qwen2.5_3b_w2v_frozen_seed2.yaml > logs_suite/qwen2.5_3b_w2v_frozen_seed2.txt 2>&1
python train.py --config configs_suite/qwen2.5_3b_w2v_frozen_seed42.yaml > logs_suite/qwen2.5_3b_w2v_frozen_seed42.txt 2>&1
python train.py --config configs_suite/qwen2.5_3b_w2v_unfrozen_seed1234.yaml > logs_suite/qwen2.5_3b_w2v_unfrozen_seed1234.txt 2>&1
python train.py --config configs_suite/qwen2.5_3b_w2v_unfrozen_seed2.yaml > logs_suite/qwen2.5_3b_w2v_unfrozen_seed2.txt 2>&1
python train.py --config configs_suite/qwen2.5_3b_w2v_unfrozen_seed42.yaml > logs_suite/qwen2.5_3b_w2v_unfrozen_seed42.txt 2>&1
python train.py --config configs_suite/qwen2.5_3b_whisperv3_frozen_seed1234.yaml > logs_suite/qwen2.5_3b_whisperv3_frozen_seed1234.txt 2>&1
python train.py --config configs_suite/qwen2.5_3b_whisperv3_frozen_seed2.yaml > logs_suite/qwen2.5_3b_whisperv3_frozen_seed2.txt 2>&1
python train.py --config configs_suite/qwen2.5_3b_whisperv3_frozen_seed42.yaml > logs_suite/qwen2.5_3b_whisperv3_frozen_seed42.txt 2>&1
python train.py --config configs_suite/qwen2.5_3b_whisperv3_unfrozen_seed1234.yaml > logs_suite/qwen2.5_3b_whisperv3_unfrozen_seed1234.txt 2>&1
python train.py --config configs_suite/qwen2.5_3b_whisperv3_unfrozen_seed2.yaml > logs_suite/qwen2.5_3b_whisperv3_unfrozen_seed2.txt 2>&1
python train.py --config configs_suite/qwen2.5_3b_whisperv3_unfrozen_seed42.yaml > logs_suite/qwen2.5_3b_whisperv3_unfrozen_seed42.txt 2>&1
python train.py --config configs_suite/qwen2.5_7b_w2v_frozen_seed1234.yaml > logs_suite/qwen2.5_7b_w2v_frozen_seed1234.txt 2>&1
python train.py --config configs_suite/qwen2.5_7b_w2v_frozen_seed2.yaml > logs_suite/qwen2.5_7b_w2v_frozen_seed2.txt 2>&1
python train.py --config configs_suite/qwen2.5_7b_w2v_frozen_seed42.yaml > logs_suite/qwen2.5_7b_w2v_frozen_seed42.txt 2>&1
python train.py --config configs_suite/qwen2.5_7b_w2v_unfrozen_seed1234.yaml > logs_suite/qwen2.5_7b_w2v_unfrozen_seed1234.txt 2>&1
python train.py --config configs_suite/qwen2.5_7b_w2v_unfrozen_seed2.yaml > logs_suite/qwen2.5_7b_w2v_unfrozen_seed2.txt 2>&1
python train.py --config configs_suite/qwen2.5_7b_w2v_unfrozen_seed42.yaml > logs_suite/qwen2.5_7b_w2v_unfrozen_seed42.txt 2>&1
python train.py --config configs_suite/qwen2.5_7b_whisperv3_frozen_seed1234.yaml > logs_suite/qwen2.5_7b_whisperv3_frozen_seed1234.txt 2>&1
python train.py --config configs_suite/qwen2.5_7b_whisperv3_frozen_seed2.yaml > logs_suite/qwen2.5_7b_whisperv3_frozen_seed2.txt 2>&1
python train.py --config configs_suite/qwen2.5_7b_whisperv3_frozen_seed42.yaml > logs_suite/qwen2.5_7b_whisperv3_frozen_seed42.txt 2>&1
python train.py --config configs_suite/qwen2.5_7b_whisperv3_unfrozen_seed1234.yaml > logs_suite/qwen2.5_7b_whisperv3_unfrozen_seed1234.txt 2>&1
python train.py --config configs_suite/qwen2.5_7b_whisperv3_unfrozen_seed2.yaml > logs_suite/qwen2.5_7b_whisperv3_unfrozen_seed2.txt 2>&1
python train.py --config configs_suite/qwen2.5_7b_whisperv3_unfrozen_seed42.yaml > logs_suite/qwen2.5_7b_whisperv3_unfrozen_seed42.txt 2>&1
