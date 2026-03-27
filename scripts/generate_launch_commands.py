import os
import glob

# Configuration
config_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/configs_prompt_lora_additional"
checkpoint_base_dir = "/ds-slt/audio/yelkheir/suite_prompt/suite_prompt"
output_script_path = "launch_lora_additional.sh"

def generate_launch_script():
    commands = []
    
    # Get all generated configs
    config_files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    
    for config_path in config_files:
        filename = os.path.basename(config_path)
        # Example filename: qwen2.5_0.5b_wav2vec2_unfrozen_seed2_prompt_alpha16.yaml
        
        parts = filename.split('_')
        
        # Parse components
        # qwen2.5_0.5b -> qwen2.5-0.5b
        model_part = f"{parts[0]}_{parts[1]}" # qwen2.5_0.5b
        model_name = model_part.replace('_', '-') # qwen2.5-0.5b
        
        encoder = parts[2] # wav2vec2 or whisperv3
        
        # seed extraction
        # parts[4] is seedX
        seed_part = parts[4]
        if not seed_part.startswith("seed"):
             # Fallback parsing if needed, but structure seems consistent
             pass
        
        # Construct checkpoint folder name
        # Pattern: qwen2.5-0.5b-instruct_{encoder}_unfrozen_{seed}
        checkpoint_name = f"{model_name}-instruct_{encoder}_unfrozen_{seed_part}"
        checkpoint_path = os.path.join(checkpoint_base_dir, checkpoint_name)
        
        # Check if checkpoint exists
        if not os.path.isdir(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path} for config {filename}")
            # continue # strictly skip or allow if maybe just not visible? User said to create them, so assume they exist or should exist.
            # But let's assume valid paths based on listing.
        
        # Log file path
        # Same name as config but .txt
        log_path = config_path.replace('.yaml', '.txt')
        
        # Construct command
        # sh run_further.sh --config {config} --pretrained_checkpoint {ckpt} &> {log} &
        cmd = f"sh run_further.sh --config {config_path} --pretrained_checkpoint {checkpoint_path}/ &> {log_path} &"
        commands.append(cmd)

    # Write to file
    with open(output_script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        for cmd in commands:
            f.write(f"{cmd}\n")
            
    print(f"Generated {len(commands)} commands in {output_script_path}")

if __name__ == "__main__":
    generate_launch_script()
