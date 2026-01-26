import os
import yaml
import copy

# Define parameters
seeds = [2, 42, 1234]
freeze_options = [True, False]
models = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct"
]

# Base templates based on provided files
w2v_template = {
    "data": {
        "train_path": "data/asv19_train.jsonl",
        "val_path": "data/asv19_val.jsonl",
        "prompt_path": "prompts/train_prompt.json"
    },
    "train": {
        "batch_size": 4,
        "epochs": 5,
        "lr": 1.0e-05,
        "lr_adapter": 1.0e-05,
        "lr_audio": 1.0e-06,
        "lr_lora": 1.0e-05,
        "num_workers": 4,
        "seed": 42,
        "grad_accum_steps": 1
    },
    "model": {
        "llm_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "llm_type": "qwen2",
        "audio_encoder_path": "/netscratch/yelkheir/ssl_models/xlsr2_300m.pt",
        "audio_encoder_type": "wav2vec2",
        "audio_encoder_source": "fairseq",
        "freeze_audio_encoder": False,
        "lora": True,
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "special_tokens": []
    },
    "output_dir": ""
}

whisper_template = {
    "data": {
        "train_path": "data/asv19_train.jsonl",
        "val_path": "data/asv19_val.jsonl",
        "prompt_path": "prompts/train_prompt.json"
    },
    "train": {
        "batch_size": 4,
        "epochs": 5,
        "lr": 1.0e-05,
        "lr_adapter": 1.0e-05,
        "lr_audio": 1.0e-06,
        "lr_lora": 1.0e-05,
        "num_workers": 4,
        "seed": 42,
        "grad_accum_steps": 1
    },
    "model": {
        "llm_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "llm_type": "qwen2",
        "audio_encoder_path": "openai/whisper-large-v3",
        "audio_encoder_type": "whisper",
        "audio_encoder_source": "huggingface",
        "freeze_audio_encoder": False,
        "lora": True,
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "special_tokens": []
    },
    "output_dir": ""
}

whisper_ner_template = {
    "data": {
        "train_path": "data/asv19_train.jsonl",
        "val_path": "data/asv19_val.jsonl",
        "prompt_path": "prompts/train_prompt.json"
    },
    "train": {
        "batch_size": 4,
        "epochs": 5,
        "lr": 1.0e-05,
        "lr_adapter": 1.0e-05,
        "lr_audio": 1.0e-06,
        "lr_lora": 1.0e-05,
        "num_workers": 4,
        "seed": 42,
        "grad_accum_steps": 1
    },
    "model": {
        "llm_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "llm_type": "qwen2",
        "audio_encoder_path": "/netscratch/yelkheir/ssl_models/whisperv3_ner.pt",
        "audio_encoder_type": "whisper_ner",
        "audio_encoder_source": "huggingface",
        "freeze_audio_encoder": False,
        "lora": True,
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "special_tokens": []
    },
    "output_dir": ""
}

eat_template = {
    "data": {
        "train_path": "data/asv19_train.jsonl",
        "val_path": "data/asv19_val.jsonl",
        "prompt_path": "prompts/train_prompt.json"
    },
    "train": {
        "batch_size": 4,
        "epochs": 5,
        "lr": 1.0e-05,
        "lr_adapter": 1.0e-05,
        "lr_audio": 1.0e-06,
        "lr_lora": 1.0e-05,
        "num_workers": 4,
        "seed": 42,
        "grad_accum_steps": 1
    },
    "model": {
        "llm_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "llm_type": "qwen2",
        "audio_encoder_path": "worstchan/EAT-large_epoch20_pretrain",
        "audio_encoder_type": "eat",
        "audio_encoder_source": "huggingface",
        "freeze_audio_encoder": False,
        "lora": True,
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "special_tokens": []
    },
    "output_dir": ""
}

output_config_dir = "configs_suite2"
os.makedirs(output_config_dir, exist_ok=True)

def get_model_short_name(model_path):
    # Extracts "0.5b", "3b", "7b" from the path
    parts = model_path.split("-")
    for part in parts:
        if part.upper().endswith("B"):
            return part.lower()
    return "unknown"

configs_created = []

for model in models:
    model_short = get_model_short_name(model)
    
    # Determine batch size based on model size
    if "3b" in model_short or "7b" in model_short:
        batch_size = 2
    else:
        batch_size = 4
    
    for seed in seeds:
        for freeze in freeze_options:
            frozen_str = "frozen" if freeze else "unfrozen"
            
            # Whisper NER Config
            whisper_ner_config = copy.deepcopy(whisper_ner_template)
            whisper_ner_config["train"]["seed"] = seed
            whisper_ner_config["train"]["batch_size"] = batch_size
            whisper_ner_config["model"]["llm_path"] = model
            whisper_ner_config["model"]["freeze_audio_encoder"] = freeze
            
            whisper_ner_output_dir = f"outputs/suite2/qwen2.5-{model_short}-instruct_whisperv3_ner_{frozen_str}_seed{seed}"
            whisper_ner_config["output_dir"] = whisper_ner_output_dir
            
            whisper_ner_filename = f"qwen2.5_{model_short}_whisperv3_ner_{frozen_str}_seed{seed}.yaml"
            whisper_ner_filepath = os.path.join(output_config_dir, whisper_ner_filename)
            
            with open(whisper_ner_filepath, "w") as f:
                yaml.dump(whisper_ner_config, f, sort_keys=False)
            configs_created.append(whisper_ner_filepath)

            # EAT Config
            eat_config = copy.deepcopy(eat_template)
            eat_config["train"]["seed"] = seed
            eat_config["train"]["batch_size"] = batch_size
            eat_config["model"]["llm_path"] = model
            eat_config["model"]["freeze_audio_encoder"] = freeze
            
            eat_output_dir = f"outputs/suite2/qwen2.5-{model_short}-instruct_eat_{frozen_str}_seed{seed}"
            eat_config["output_dir"] = eat_output_dir
            
            eat_filename = f"qwen2.5_{model_short}_eat_{frozen_str}_seed{seed}.yaml"
            eat_filepath = os.path.join(output_config_dir, eat_filename)
            
            with open(eat_filepath, "w") as f:
                yaml.dump(eat_config, f, sort_keys=False)
            configs_created.append(eat_filepath)

print(f"Created {len(configs_created)} config files in {output_config_dir}")
