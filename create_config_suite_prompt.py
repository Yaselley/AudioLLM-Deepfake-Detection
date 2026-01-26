"""
Create a suite of prompt configs for configs_prompt/.
Similar to qwen2.5_0.5b_whisperv3_unfrozen_seed42_prompt.yaml.
Varies: LLM (0.5B, 3B, 7B), audio encoder (whisper, wav2vec2, eat, whisper_ner),
        freeze (frozen/unfrozen), seed (42, 2, 1234).
"""
import os
import yaml
import copy

OUTPUT_DIR = "configs_prompt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA = {
    "train_path": "data/asv19_train.jsonl",
    "val_path": "data/asv19_val.jsonl",
    "prompt_path": "prompts/train_prompt.json",
}

TRAIN_BASE = {
    "batch_size": 4,
    "epochs": 5,
    "lr": 1.0e-05,
    "lr_adapter": 1.0e-05,
    "lr_audio": 1.0e-06,
    "lr_lora": 1.0e-05,
    "num_workers": 4,
    "seed": 42,
    "grad_accum_steps": 1,
}

MODEL_BASE = {
    "llm_path": "Qwen/Qwen2.5-0.5B-Instruct",
    "llm_type": "qwen2",
    "lora": False,
    "lora_rank": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "special_tokens": [],
}

ENCODERS = {
    "whisper": {
        "audio_encoder_path": "openai/whisper-large-v3",
        "audio_encoder_type": "whisper",
        "audio_encoder_source": "huggingface",
        "name": "whisperv3",
    },
    "wav2vec2": {
        "audio_encoder_path": "/netscratch/yelkheir/ssl_models/xlsr2_300m.pt",
        "audio_encoder_type": "wav2vec2",
        "audio_encoder_source": "fairseq",
        "name": "wav2vec2",
    },
    "whisper_ner": {
        "audio_encoder_path": "/netscratch/yelkheir/ssl_models/whisperv3_ner.pt",
        "audio_encoder_type": "whisper_ner",
        "audio_encoder_source": "huggingface",
        "name": "whisperv3_ner",
    },
}


    # "eat": {
    #     "audio_encoder_path": "worstchan/EAT-large_epoch20_pretrain",
    #     "audio_encoder_type": "eat",
    #     "audio_encoder_source": "huggingface",
    #     "name": "eat",
    # },

MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "0.5b"),
    ("Qwen/Qwen2.5-3B-Instruct", "3b"),
    ("Qwen/Qwen2.5-7B-Instruct", "7b"),
]

SEEDS = [42, 2, 1234]
FREEZE_OPTS = [True, False]


def build_config(llm_path, model_short, enc_key, freeze, seed):
    enc = ENCODERS[enc_key]
    batch_size = 4 if model_short == "0.5b" else 2
    frozen_str = "frozen" if freeze else "unfrozen"
    enc_name = enc["name"]

    cfg = {
        "data": copy.deepcopy(DATA),
        "train": copy.deepcopy(TRAIN_BASE),
        "model": copy.deepcopy(MODEL_BASE),
        "output_dir": "",
    }
    cfg["train"]["seed"] = seed
    cfg["train"]["batch_size"] = batch_size
    cfg["model"]["llm_path"] = llm_path
    cfg["model"]["audio_encoder_path"] = enc["audio_encoder_path"]
    cfg["model"]["audio_encoder_type"] = enc["audio_encoder_type"]
    cfg["model"]["audio_encoder_source"] = enc["audio_encoder_source"]
    cfg["model"]["freeze_audio_encoder"] = freeze
    cfg["output_dir"] = (
        f"outputs/suite_prompt/qwen2.5-{model_short}-instruct_{enc_name}_{frozen_str}_seed{seed}"
    )
    return cfg, enc_name, frozen_str


def main():
    created = []
    for llm_path, model_short in MODELS:
        for enc_key in ENCODERS:
            for freeze in FREEZE_OPTS:
                for seed in SEEDS:
                    cfg, enc_name, frozen_str = build_config(
                        llm_path, model_short, enc_key, freeze, seed
                    )
                    fname = (
                        f"qwen2.5_{model_short}_{enc_name}_{frozen_str}_seed{seed}_prompt.yaml"
                    )
                    path = os.path.join(OUTPUT_DIR, fname)
                    with open(path, "w") as f:
                        yaml.dump(cfg, f, sort_keys=False)
                    created.append(path)

    print(f"Created {len(created)} configs in {OUTPUT_DIR}/")
    for p in sorted(created):
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
