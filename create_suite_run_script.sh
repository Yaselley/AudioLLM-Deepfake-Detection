#!/bin/bash

# Script to generate run_suite.sh with all experiment commands for the suite

cd /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection

# Create run_suite.sh file
RUN_FILE="run_suite.sh"
echo "#!/bin/bash" > "$RUN_FILE"
echo "" >> "$RUN_FILE"
echo "cd /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection" >> "$RUN_FILE"
echo "mkdir -p logs_suite" >> "$RUN_FILE"
echo "" >> "$RUN_FILE"

# Find all config files in configs_suite directory and sort them
for config in $(ls configs_suite/*.yaml | sort); do
    # Extract name from config path
    name=$(basename "$config" .yaml)
    
    # Generate command: python train.py --config <config> > logs_suite/<name>.txt
    echo "python train.py --config $config > logs_suite/${name}.txt 2>&1" >> "$RUN_FILE"
done

# Make run_suite.sh executable
chmod +x "$RUN_FILE"

echo "Generated $RUN_FILE with $(grep -c "python train.py" "$RUN_FILE") experiment commands"
echo "To run all experiments, execute: ./$RUN_FILE"
