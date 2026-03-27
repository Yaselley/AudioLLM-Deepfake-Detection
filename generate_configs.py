import os
import yaml

base_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/configs_opensmile_lora_additional"
os.makedirs(base_dir, exist_ok=True)

models = {
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct"
}

seeds = [2, 42, 1234]

audio_encoders = {
    "whisper": {
        "path": "openai/whisper-large-v3",
        "type": "whisper",
        "source": "huggingface",
        "output_prefix": "whisper"
    },
    "wav2vec2": {
        "path": "/netscratch/yelkheir/ssl_models/xlsr2_300m.pt",
        "type": "wav2vec2",
        "source": "fairseq",
        "output_prefix": "w2v"
    }
}

template_str = """data:
  train_path: data/asv19_train.jsonl
  val_path: data/asv19_val.jsonl
  prompt_path: prompts/train_prompt_opensmile_after.json
train:
  batch_size: 2
  epochs: 5
  lr: 1e-05
  lr_adapter: 1e-05
  lr_audio: 1e-06
  lr_lora: 1e-05
  num_workers: 4
  seed: {seed}
  grad_accum_steps: 1
model:
  llm_path: {llm_path}
  llm_type: qwen2
  audio_encoder_path: {audio_encoder_path}
  audio_encoder_type: {audio_encoder_type}
  audio_encoder_source: {audio_encoder_source}
  freeze_audio_encoder: false
  lora: true
  lora_rank: 64
  lora_alpha: 16
  lora_dropout: 0.05
  special_tokens: []
  use_opensmile: true
  opensmile_only: false
output_dir: outputs/opensmile_lora_additional/{output_prefix}_{model_size}_lora_stage2_after_{seed}_alpha16
"""

for model_size, llm_path in models.items():
    for enc_name, enc_info in audio_encoders.items():
        for seed in seeds:
            filename = f"qwen2.5_{model_size}_{enc_name}_opensmile_seed{seed}_alpha16.yaml"
            filepath = os.path.join(base_dir, filename)
            
            content = template_str.format(
                seed=seed,
                llm_path=llm_path,
                audio_encoder_path=enc_info["path"],
                audio_encoder_type=enc_info["type"],
                audio_encoder_source=enc_info["source"],
                output_prefix=enc_info["output_prefix"],
                model_size=model_size
            )
            
            with open(filepath, "w") as f:
                f.write(content)
            print(f"Created {filepath}")
