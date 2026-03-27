import os
import glob

# Configuration
config_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/configs_opensmile_lora_additional_before"
checkpoint_base_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/opensmile/before"
output_script_path = "launch_opensmile_lora_additional_before.sh"

def generate_launch_script():
    commands = []
    
    # Get all generated configs
    config_files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    
    for config_path in config_files:
        filename = os.path.basename(config_path)
        # Example filename: qwen2.5_0.5b_wav2vec2_opensmile_seed2_alpha16.yaml
        
        parts = filename.split('_')
        
        if "wav2vec2" in filename:
             encoder_prefix = "w2v"
        elif "whisper" in filename:
             encoder_prefix = "whisper"
        else:
             print(f"Skipping unknown encoder in {filename}")
             continue

        # Extract model size
        # parts[1] is typically the size (0.5b)
        model_size_str = parts[1]
        
        # Map filename size to folder size convention
        if model_size_str == "0.5b":
            folder_size = "0.5"
        else:
            folder_size = model_size_str
             
        # seed extraction
        # Filename has ..._seed{seed}_alpha...
        seed_part = next((p for p in parts if p.startswith("seed")), None)
        if not seed_part:
            print(f"Skipping {filename}: could not find seed")
            continue
            
        seed = seed_part.replace("seed", "")
        
        # Checkpoint folder construction:
        # Based on listing, it seems like: {encoder}_{size}_nolora_before_{seed}
        checkpoint_folder_name = f"{encoder_prefix}_{folder_size}_nolora_before_{seed}"
        checkpoint_path = os.path.join(checkpoint_base_dir, checkpoint_folder_name)
        
        if not os.path.isdir(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path} for config {filename}")
            
        # Log file path
        # Same name as config but .txt, in the config dir
        log_path = config_path.replace('.yaml', '.txt')
        
        # Construct command
        script_path = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/train_tokens_opensmile_stage2.py"
        
        # Note: Using absolute paths
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
