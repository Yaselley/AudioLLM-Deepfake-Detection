import os
import yaml
import glob
import re

# Configuration
input_base_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/opensmile/after"
output_config_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/configs_opensmile_lora_additional"
seeds = [2, 42, 1234]
models = ["qwen2.5_0.5b"] # Logic will filter based on folder name
encoders = ["w2v", "whisper"] # Folder prefixes
alphas = [16, 64, 128, 256]

os.makedirs(output_config_dir, exist_ok=True)

def generate_configs():
    # List all subdirectories in the input base dir
    # We are looking for folders like: w2v_0.5_nolora_after_2, whisper_0.5_nolora_after_1234
    
    subdirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
    
    for subdir in subdirs:
        # Filter for relevant folders:
        # 1. Must be 0.5 (Qwen 0.5B)
        # 2. Must be nolora (since we want to use pretrained frozen checkpoint)
        # 3. Must be w2v or whisper
        
        if "_0.5_" not in subdir:
            continue
        if "_nolora_" not in subdir:
            continue
        
        is_w2v = subdir.startswith("w2v_")
        is_whisper = subdir.startswith("whisper_")
        
        if not (is_w2v or is_whisper):
            continue
            
        encoder_name = "wav2vec2" if is_w2v else "whisper" # For filename purposes
        
        # Extract seed from folder name (last part)
        try:
            seed = int(subdir.split('_')[-1])
        except ValueError:
            print(f"Skipping {subdir}: could not extract seed")
            continue
            
        if seed not in seeds:
            continue

        # Found a valid source folder
        source_dir = os.path.join(input_base_dir, subdir)
        config_path = os.path.join(source_dir, "config.json")
        
        if not os.path.exists(config_path):
            print(f"Warning: config.json not found in {source_dir}")
            continue
            
        with open(config_path, 'r') as f:
            base_config = yaml.safe_load(f) # json is valid yaml
            
        for alpha in alphas:
            new_config = base_config.copy()
            
            # Update LoRA settings
            if "model" not in new_config:
                 print(f"Error: 'model' key missing in {config_path}")
                 continue
                 
            new_config['model']['lora'] = True
            new_config['model']['lora_rank'] = 64
            new_config['model']['lora_alpha'] = alpha
            
            # Ensure opensmile flags are preserved (should be in base config, but just in case)
            # new_config['model']['use_opensmile'] = True 
            
            # Update output directory
            # Original: outputs/opensmile/after/w2v_0.5_nolora_after_2
            # New: outputs/opensmile_lora_additional/w2v_0.5_nolora_after_2_alpha{alpha}
            # Note: keeping "nolora" in name might be confusing, maybe replace with "lora_stage2"?
            # Let's replace "nolora" with "lora_stage2"
            
            dirname = subdir.replace("_nolora_", "_lora_stage2_")
            new_output_dir = os.path.join("outputs", "opensmile_lora_additional", f"{dirname}_alpha{alpha}")
            new_config['output_dir'] = new_output_dir
            
            # Save new config
            # Filename: qwen2.5_0.5b_{encoder}_opensmile_seed{seed}_alpha{alpha}.yaml
            new_filename = f"qwen2.5_0.5b_{encoder_name}_opensmile_seed{seed}_alpha{alpha}.yaml"
            new_filepath = os.path.join(output_config_dir, new_filename)
            
            with open(new_filepath, 'w') as f:
                yaml.dump(new_config, f, sort_keys=False, default_flow_style=False)
            
            print(f"Generated config: {new_filepath}")
            # print(f"  Source checkpoint will be: {source_dir}")

if __name__ == "__main__":
    generate_configs()
