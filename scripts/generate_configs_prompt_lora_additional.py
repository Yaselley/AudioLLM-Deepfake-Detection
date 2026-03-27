import os
import yaml

# Configuration
base_config_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/configs_prompt"
output_config_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/configs_prompt_lora_additional"
seeds = [2, 42, 1234]
models = ["qwen2.5_0.5b"]  # Only Qwen 0.5B
encoders = ["wav2vec2", "whisperv3"] # Wav2Vec2 and Whisper (excluding EAT/NER for now based on request "w2v and whisper")
alphas = [16, 64, 128] # Alpha values to try. Rank will stay 64 (default) or can be varied if needed.
# Usually alpha is scaling factor. If rank is 64, alpha 16 (scaling 16/64=0.25), 64 (1.0), 128 (2.0)

os.makedirs(output_config_dir, exist_ok=True)

def generate_configs():
    for model_prefix in models:
        for encoder in encoders:
            for seed in seeds:
                # Construct base filename pattern
                # Example: qwen2.5_0.5b_wav2vec2_unfrozen_seed2_prompt.yaml
                # Note: Whisper files are named whisperv3_unfrozen... (without ner based on list)
                
                base_filename = f"{model_prefix}_{encoder}_unfrozen_seed{seed}_prompt.yaml"
                base_filepath = os.path.join(base_config_dir, base_filename)
                
                if not os.path.exists(base_filepath):
                    print(f"Warning: Base config not found: {base_filepath}")
                    continue
                
                with open(base_filepath, 'r') as f:
                    config = yaml.safe_load(f)
                
                for alpha in alphas:
                    new_config = config.copy()
                    
                    # Update LoRA settings
                    new_config['model']['lora'] = True
                    new_config['model']['lora_rank'] = 64 # Keep default rank 64? User said "alpha should be different", usually implies rank is fixed.
                    new_config['model']['lora_alpha'] = alpha
                    
                    # Update output directory
                    # Original: outputs/suite_prompt/qwen2.5-0.5b-instruct_wav2vec2_unfrozen_seed2
                    # New: outputs/suite_prompt_lora_additional/qwen2.5-0.5b-instruct_wav2vec2_unfrozen_seed2_alpha{alpha}
                    
                    original_output_dir = new_config['output_dir']
                    # Replace suite_prompt with suite_prompt_lora_additional if present, or just append
                    if "suite_prompt" in original_output_dir:
                        new_output_dir = original_output_dir.replace("suite_prompt", "suite_prompt_lora_additional")
                    else:
                        new_output_dir = os.path.join("outputs", "suite_prompt_lora_additional", os.path.basename(original_output_dir))
                        
                    new_output_dir = f"{new_output_dir}_alpha{alpha}"
                    new_config['output_dir'] = new_output_dir
                    
                    # Save new config
                    new_filename = f"{model_prefix}_{encoder}_unfrozen_seed{seed}_prompt_alpha{alpha}.yaml"
                    new_filepath = os.path.join(output_config_dir, new_filename)
                    
                    with open(new_filepath, 'w') as f:
                        yaml.dump(new_config, f, sort_keys=False, default_flow_style=False)
                    
                    print(f"Generated: {new_filepath}")

if __name__ == "__main__":
    generate_configs()
