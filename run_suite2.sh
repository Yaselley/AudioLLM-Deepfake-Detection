#!/bin/bash

cd /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection
mkdir -p logs_suite2

# First generate the configs
python create_config_suite2.py

# Qwen 0.5B
# Whisper NER
python train.py --config configs_suite2/qwen2.5_0.5b_whisperv3_ner_frozen_seed1234.yaml > logs_suite2/qwen2.5_0.5b_whisperv3_ner_frozen_seed1234.txt 2>&1
python train.py --config configs_suite2/qwen2.5_0.5b_whisperv3_ner_frozen_seed2.yaml > logs_suite2/qwen2.5_0.5b_whisperv3_ner_frozen_seed2.txt 2>&1
python train.py --config configs_suite2/qwen2.5_0.5b_whisperv3_ner_frozen_seed42.yaml > logs_suite2/qwen2.5_0.5b_whisperv3_ner_frozen_seed42.txt 2>&1
python train.py --config configs_suite2/qwen2.5_0.5b_whisperv3_ner_unfrozen_seed1234.yaml > logs_suite2/qwen2.5_0.5b_whisperv3_ner_unfrozen_seed1234.txt 2>&1
python train.py --config configs_suite2/qwen2.5_0.5b_whisperv3_ner_unfrozen_seed2.yaml > logs_suite2/qwen2.5_0.5b_whisperv3_ner_unfrozen_seed2.txt 2>&1
python train.py --config configs_suite2/qwen2.5_0.5b_whisperv3_ner_unfrozen_seed42.yaml > logs_suite2/qwen2.5_0.5b_whisperv3_ner_unfrozen_seed42.txt 2>&1

# EAT
python train.py --config configs_suite2/qwen2.5_0.5b_eat_frozen_seed1234.yaml > logs_suite2/qwen2.5_0.5b_eat_frozen_seed1234.txt 2>&1
python train.py --config configs_suite2/qwen2.5_0.5b_eat_frozen_seed2.yaml > logs_suite2/qwen2.5_0.5b_eat_frozen_seed2.txt 2>&1
python train.py --config configs_suite2/qwen2.5_0.5b_eat_frozen_seed42.yaml > logs_suite2/qwen2.5_0.5b_eat_frozen_seed42.txt 2>&1
python train.py --config configs_suite2/qwen2.5_0.5b_eat_unfrozen_seed1234.yaml > logs_suite2/qwen2.5_0.5b_eat_unfrozen_seed1234.txt 2>&1
python train.py --config configs_suite2/qwen2.5_0.5b_eat_unfrozen_seed2.yaml > logs_suite2/qwen2.5_0.5b_eat_unfrozen_seed2.txt 2>&1
python train.py --config configs_suite2/qwen2.5_0.5b_eat_unfrozen_seed42.yaml > logs_suite2/qwen2.5_0.5b_eat_unfrozen_seed42.txt 2>&1

# Qwen 3B
# Whisper NER
python train.py --config configs_suite2/qwen2.5_3b_whisperv3_ner_frozen_seed1234.yaml > logs_suite2/qwen2.5_3b_whisperv3_ner_frozen_seed1234.txt 2>&1
python train.py --config configs_suite2/qwen2.5_3b_whisperv3_ner_frozen_seed2.yaml > logs_suite2/qwen2.5_3b_whisperv3_ner_frozen_seed2.txt 2>&1
python train.py --config configs_suite2/qwen2.5_3b_whisperv3_ner_frozen_seed42.yaml > logs_suite2/qwen2.5_3b_whisperv3_ner_frozen_seed42.txt 2>&1
python train.py --config configs_suite2/qwen2.5_3b_whisperv3_ner_unfrozen_seed1234.yaml > logs_suite2/qwen2.5_3b_whisperv3_ner_unfrozen_seed1234.txt 2>&1
python train.py --config configs_suite2/qwen2.5_3b_whisperv3_ner_unfrozen_seed2.yaml > logs_suite2/qwen2.5_3b_whisperv3_ner_unfrozen_seed2.txt 2>&1
python train.py --config configs_suite2/qwen2.5_3b_whisperv3_ner_unfrozen_seed42.yaml > logs_suite2/qwen2.5_3b_whisperv3_ner_unfrozen_seed42.txt 2>&1

# EAT
python train.py --config configs_suite2/qwen2.5_3b_eat_frozen_seed1234.yaml > logs_suite2/qwen2.5_3b_eat_frozen_seed1234.txt 2>&1
python train.py --config configs_suite2/qwen2.5_3b_eat_frozen_seed2.yaml > logs_suite2/qwen2.5_3b_eat_frozen_seed2.txt 2>&1
python train.py --config configs_suite2/qwen2.5_3b_eat_frozen_seed42.yaml > logs_suite2/qwen2.5_3b_eat_frozen_seed42.txt 2>&1
python train.py --config configs_suite2/qwen2.5_3b_eat_unfrozen_seed1234.yaml > logs_suite2/qwen2.5_3b_eat_unfrozen_seed1234.txt 2>&1
python train.py --config configs_suite2/qwen2.5_3b_eat_unfrozen_seed2.yaml > logs_suite2/qwen2.5_3b_eat_unfrozen_seed2.txt 2>&1
python train.py --config configs_suite2/qwen2.5_3b_eat_unfrozen_seed42.yaml > logs_suite2/qwen2.5_3b_eat_unfrozen_seed42.txt 2>&1

# Qwen 7B
# Whisper NER
python train.py --config configs_suite2/qwen2.5_7b_whisperv3_ner_frozen_seed1234.yaml > logs_suite2/qwen2.5_7b_whisperv3_ner_frozen_seed1234.txt 2>&1
python train.py --config configs_suite2/qwen2.5_7b_whisperv3_ner_frozen_seed2.yaml > logs_suite2/qwen2.5_7b_whisperv3_ner_frozen_seed2.txt 2>&1
python train.py --config configs_suite2/qwen2.5_7b_whisperv3_ner_frozen_seed42.yaml > logs_suite2/qwen2.5_7b_whisperv3_ner_frozen_seed42.txt 2>&1
python train.py --config configs_suite2/qwen2.5_7b_whisperv3_ner_unfrozen_seed1234.yaml > logs_suite2/qwen2.5_7b_whisperv3_ner_unfrozen_seed1234.txt 2>&1
python train.py --config configs_suite2/qwen2.5_7b_whisperv3_ner_unfrozen_seed2.yaml > logs_suite2/qwen2.5_7b_whisperv3_ner_unfrozen_seed2.txt 2>&1
python train.py --config configs_suite2/qwen2.5_7b_whisperv3_ner_unfrozen_seed42.yaml > logs_suite2/qwen2.5_7b_whisperv3_ner_unfrozen_seed42.txt 2>&1

# EAT
python train.py --config configs_suite2/qwen2.5_7b_eat_frozen_seed1234.yaml > logs_suite2/qwen2.5_7b_eat_frozen_seed1234.txt 2>&1
python train.py --config configs_suite2/qwen2.5_7b_eat_frozen_seed2.yaml > logs_suite2/qwen2.5_7b_eat_frozen_seed2.txt 2>&1
python train.py --config configs_suite2/qwen2.5_7b_eat_frozen_seed42.yaml > logs_suite2/qwen2.5_7b_eat_frozen_seed42.txt 2>&1
python train.py --config configs_suite2/qwen2.5_7b_eat_unfrozen_seed1234.yaml > logs_suite2/qwen2.5_7b_eat_unfrozen_seed1234.txt 2>&1
python train.py --config configs_suite2/qwen2.5_7b_eat_unfrozen_seed2.yaml > logs_suite2/qwen2.5_7b_eat_unfrozen_seed2.txt 2>&1
python train.py --config configs_suite2/qwen2.5_7b_eat_unfrozen_seed42.yaml > logs_suite2/qwen2.5_7b_eat_unfrozen_seed42.txt 2>&1
