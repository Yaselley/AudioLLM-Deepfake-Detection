import os
import yaml
import argparse

def generate_configs(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    seeds = [1234, 42, 2]
    
    if args.model_prefix:
        prefixes = [args.model_prefix]
    else:
        prefixes = ["w2v", "whisper"]

    patterns = []
    for prefix in prefixes:
        patterns.extend([
            f"{prefix}_0.5_lora_after",
            f"{prefix}_0.5_lora_before",
            f"{prefix}_0.5_nolora_after",
            f"{prefix}_0.5_nolora_before"
        ])
    
    base_config_dir = args.base_config_dir
    
    count = 0
    
    for pattern in patterns:
        for seed in seeds:
            base_filename = f"{pattern}_{seed}.yaml"
            base_path = os.path.join(base_config_dir, base_filename)
            
            if not os.path.exists(base_path):
                print(f"MISSING: {base_path}")
                continue
            
            print(f"Processing: {base_filename}")
            with open(base_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config["model"]["llm_path"] = "Qwen/Qwen2.5-0.5B-Instruct"
            config["model"]["llm_type"] = "qwen2"
            
            if "nolora" not in pattern:
                config["model"]["lora"] = True
                config["model"]["lora_rank"] = args.lora_rank
                config["model"]["lora_alpha"] = args.lora_alpha
            else:
                config["model"]["lora"] = False
            
            config["model"]["use_opensmile"] = True
            
            old_output_dir = config.get("output_dir", "")
            if "opensmile" in old_output_dir:
                 new_output_dir = old_output_dir.replace("opensmile", "opensmile_selected")
            else:
                 # Fallback if path structure is different
                 new_output_dir = os.path.join("outputs", "opensmile_selected", base_filename.replace(".yaml", ""))
            
            config["output_dir"] = new_output_dir
            
            new_filename = os.path.join(output_dir, base_filename)
            with open(new_filename, 'w') as f:
                yaml.dump(config, f, sort_keys=False)
            
            count += 1

    print(f"Generated {count} configs in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--opensmile_selected", action="store_true")
    parser.add_argument("--model_prefix", type=str, default=None, help="Prefix for config files (e.g., w2v, whisper)")
    parser.add_argument("--llm_types", type=str, default="0.5b") 
    
    args = parser.parse_args()
    generate_configs(args)
