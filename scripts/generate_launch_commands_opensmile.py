import os
import glob

# Configuration
config_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/configs_opensmile_lora_additional"
checkpoint_base_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/opensmile/after"
output_script_path = "launch_opensmile_lora_additional.sh"

def generate_launch_script():
    commands = []
    
    # Get all generated configs
    config_files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    
    # Filter for specific new configs (3b, 7b, alpha16)
    filtered_configs = []
    for f in config_files:
        if "alpha16" in f and ("3b" in f or "7b" in f):
            filtered_configs.append(f)
            
    for config_path in filtered_configs:
        filename = os.path.basename(config_path)
        # Example filename: qwen2.5_0.5b_wav2vec2_opensmile_seed2_alpha16.yaml
        # Example filename: qwen2.5_3b_whisper_opensmile_seed2_alpha16.yaml
        
        parts = filename.split('_')
        
        # Determine original folder name based on encoder and seed
        # Pattern in checkpoint dir: {encoder}_{size}_nolora_after_{seed}
        
        if "wav2vec2" in filename:
             encoder_prefix = "w2v"
        elif "whisper" in filename:
             encoder_prefix = "whisper"
        else:
             print(f"Skipping unknown encoder in {filename}")
             continue

        # Extract model size
        # parts[1] is typically the size (0.5b, 3b, 7b)
        model_size_str = parts[1]
        
        # Map filename size to folder size convention
        if model_size_str == "0.5b":
            folder_size = "0.5"
        else:
            folder_size = model_size_str
             
        # seed extraction
        # Filename has ..._seed{seed}_alpha...
        # Find the part that starts with "seed"
        seed_part = next((p for p in parts if p.startswith("seed")), None)
        if not seed_part:
            print(f"Skipping {filename}: could not find seed")
            continue
            
        seed = seed_part.replace("seed", "")
        
        checkpoint_folder_name = f"{encoder_prefix}_{folder_size}_nolora_after_{seed}"
        checkpoint_path = os.path.join(checkpoint_base_dir, checkpoint_folder_name)
        
        if not os.path.isdir(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path} for config {filename}")
            # We continue anyway as requested, or maybe we should skip? 
            # The original script had commented out continue, so we'll warn but include it.
            
        # Log file path
        # Same name as config but .txt, in the config dir
        log_path = config_path.replace('.yaml', '.txt')
        
        # Construct command
        # python3 train_tokens_opensmile_stage2.py --config {config} --pretrained_checkpoint {ckpt} &> {log} &
        
        # Use absolute paths for everything to be safe
        script_path = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/train_tokens_opensmile_stage2.py"
        
        cmd = f"python3 {script_path} --config {config_path} --pretrained_checkpoint {checkpoint_path} > {log_path} 2>&1 &"
        commands.append(cmd)

    # Write to file
    with open(output_script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        for cmd in commands:
            f.write(f"{cmd}\n")
            
    print(f"Generated {len(commands)} commands in {output_script_path}")

if __name__ == "__main__":
    generate_launch_script()
