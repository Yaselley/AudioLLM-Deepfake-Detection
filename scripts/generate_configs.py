import os
import yaml


        # {"name": "qwen2-0.5b-instruct", "path": "Qwen/Qwen2.5-0.5B-Instruct", "type": "qwen2"}, # Closest standard is 0.5B usually
        # {"name": "qwen2-0.5b", "path": "Qwen/Qwen2.5-0.5B", "type": "qwen2"}, # Closest standard is 0.5B usually
        # {"name": "qwen2-1.5b-instruct", "path": "Qwen/Qwen2.5-1.5B-Instruct", "type": "qwen2"}, # Closest standard is 0.5B usually
        # {"name": "qwen2-1.5b", "path": "Qwen/Qwen2.5-1.5B", "type": "qwen2"}, # Closest standard is 0.5B usually
        # {"name": "qwen2-3b-instruct", "path": "Qwen/Qwen2.5-3B-Instruct", "type": "qwen2"}, # Closest standard is 0.5B usually
        # {"name": "qwen2-3b", "path": "Qwen/Qwen2.5-3B", "type": "qwen2"}, # Closest standard is 0.5B usually
        # {"name": "qwen2-7b-instruct", "path": "Qwen/Qwen2.5-7B-Instruct", "type": "qwen2"}, # Closest standard is 0.5B usually
        # {"name": "qwen2-7b", "path": "Qwen/Qwen2.5-7B", "type": "qwen2"}, # Closest standard is 0.5B usually

    #    {
    #         "name": "wav2vec2",
    #         "path": "/netscratch/yelkheir/ssl_models/xlsr2_300m.pt",
    #         "type": "wav2vec2",
    #         "source": "fairseq"
    #     },
    #     {
    #         "name": "wavlm",
    #         "path": "/netscratch/yelkheir/ssl_models/WavLM-Large.pt",
    #         "type": "wavlm",
    #         "source": "unil"
    #     },
    #     {
    #         "name": "wav2vec2-bert",
    #         "path": "facebook/w2v-bert-2.0",
    #         "type": "wav2vec2-bert",
    #         "source": "huggingface"
    #     {"name": "frozen", "val": True},

def generate_configs():
    output_dir = "configs_final"
    os.makedirs(output_dir, exist_ok=True)

    llms = [
        {"name": "qwen2-7b-instruct", "path": "Qwen/Qwen2.5-7B-Instruct", "type": "qwen2"},
        {"name": "llama-3-8b-instruct", "path": "meta-llama/Meta-Llama-3-8B-Instruct", "type": "llama"},
        {"name": "qwen2-8b-instruct", "path": "Qwen/Qwen3-8B-Instruct", "type": "qwen3"},
    ]

    encoders = [
        {
            "name": "whisper",
            "path": "openai/whisper-large-v3-turbo",
            "type": "whisper",
            "source": "huggingface"
        }
    ]

    freeze_options = [
        {"name": "finetune", "val": False}
    ]

    base_config = {
        "data": {
            "train_path": "data/asv19_train.jsonl",
            "val_path": "data/asv19_val.jsonl",
            "prompt_path": "prompts/train_prompt.json"
        },
        "train": {
            "batch_size": 2,
            "epochs": 5,
            "lr": 1e-5,
            "lr_adapter": 1e-6,
            "lr_audio": 1e-5,
            "lr_lora": 1e-5,
            "num_workers": 4,
            "seed": 42,
            "grad_accum_steps": 2
        }
    }

    for llm in llms:
        for enc in encoders:
            for freeze in freeze_options:
                config = base_config.copy()
                
                # Model Config
                config["model"] = {
                    "llm_path": llm["path"],
                    "llm_type": llm["type"],
                    "audio_encoder_path": enc["path"],
                    "audio_encoder_type": enc["type"],
                    "audio_encoder_source": enc["source"],
                    "freeze_audio_encoder": freeze["val"],
                    "lora": True,
                    "lora_rank": 64,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "special_tokens": []
                }
                
                # Output Dir
                config_name = f"{llm['name']}_{enc['name']}_{freeze['name']}"
                config["output_dir"] = f"output2/{config_name}"
                
                # Save
                filename = os.path.join(output_dir, f"{config_name}.yaml")
                with open(filename, 'w') as f:
                    yaml.dump(config, f, sort_keys=False)
                
    print(f"Generated {len(llms) * len(encoders) * len(freeze_options)} configs in {output_dir}")

if __name__ == "__main__":
    generate_configs()
