#!/bin/bash

# Script to generate run.sh with all experiment commands

cd /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection

# List of all config files
CONFIGS=(
    "configs_final/qwen1.5_wav2vec.yaml"
    "configs_final/qwen1.5_wav2vec_freeze.yaml"
    "configs_final/qwen1.5_whisper.yaml"
    "configs_final/qwen1.5_whisper_freeze.yaml"
    "configs_final/qwen3_wav2vec.yaml"
    "configs_final/qwen3_wav2vec_freeze.yaml"
    "configs_final/qwen3_whisper.yaml"
    "configs_final/qwen3_whisper_freeze.yaml"
    "configs_final/qwen7_wav2vec.yaml"
    "configs_final/qwen7_wav2vec_freeze.yaml"
    "configs_final/qwen7_whisper.yaml"
    "configs_final/qwen7_whisper_freeze.yaml"
)

# Create run.sh file
RUN_FILE="run.sh"
echo "#!/bin/bash" > "$RUN_FILE"
echo "" >> "$RUN_FILE"
echo "cd /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection" >> "$RUN_FILE"
echo "" >> "$RUN_FILE"

# Generate commands for each config
for config in "${CONFIGS[@]}"; do
    # Extract name from config path (e.g., qwen1.5_wav2vec.yaml -> qwen1.5_wav2vec)
    name=$(basename "$config" .yaml)
    
    # Generate command: python train.py --config <config> > <name>.txt
    echo "python train.py --config $config > ${name}.txt 2>&1" >> "$RUN_FILE"
done

# Make run.sh executable
chmod +x "$RUN_FILE"

echo "Generated $RUN_FILE with $(wc -l < "$RUN_FILE") lines"
echo "To run all experiments, execute: ./run.sh"
