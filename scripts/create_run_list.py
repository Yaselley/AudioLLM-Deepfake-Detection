llms = [
    "qwen2-0.5b-instruct",
    "qwen2-0.5b",
    "qwen2-1.5b-instruct",
    "qwen2-1.5b",
    "qwen2-3b-instruct",
    "qwen2-3b",
    "qwen2-7b-instruct",
    "qwen2-7b",
]

encoders_priority = [
    "whisper",
    "wav2vec2",
    "wavlm",
    "wav2vec2-bert"
]

freeze_options = [
    "frozen",
    "finetune"
]

# Ensure output/errors directory exists
# mkdir -p output/errors

commands = []

# Header
print("#!/bin/bash")
print("mkdir -p output/errors")
print("")

for enc in encoders_priority:
    for llm in llms:
        for freeze in freeze_options:
            config_name = f"{llm}_{enc}_{freeze}"
            config_path = f"/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/{config_name}.yaml"
            error_log = f"output/errors/{config_name}.txt"
            
            # Note: User provided path "AudioLLM-Deepfake-Detection/output/errors/..." in example
            # but command is likely run from /netscratch/yelkheir/DeepFense
            # Assuming relative to run.sh location which is typically parent folder?
            # Or run.sh is in DeepFense and calls AudioLLM-Deepfake-Detection code.
            
            # The example was:
            # sh run.sh --config /path/to/config &> AudioLLM-Deepfake-Detection/output/errors/name.txt &
            
            cmd = f"sh run.sh --config {config_path} &> AudioLLM-Deepfake-Detection/{error_log} &"
            # Note: User asked for "&> ... &" (background). 
            # If we run all in background at once, it will crash the machine (40 jobs).
            # I will write them sequentially or let the user decide how to dispatch.
            # User said "file txt with all experiments to be runned like this ... &"
            # So I will add the "&" at the end.
            
            print(f"{cmd}")
            # print("sleep 10") # Add small sleep to stagger starts slightly if running a batch


sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b-instruct_whisper_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b-instruct_whisper_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b-instruct_whisper_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b-instruct_whisper_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b_whisper_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b_whisper_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b_whisper_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b_whisper_finetune.txt &

sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b-instruct_whisper_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b-instruct_whisper_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b-instruct_whisper_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b-instruct_whisper_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b_whisper_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b_whisper_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b_whisper_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b_whisper_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b-instruct_whisper_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b-instruct_whisper_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b-instruct_whisper_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b-instruct_whisper_finetune.txt &

sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b-instruct_whisper_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b-instruct_whisper_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b-instruct_whisper_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b-instruct_whisper_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b_whisper_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b_whisper_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b_whisper_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b_whisper_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b_whisper_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b_whisper_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b_whisper_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b_whisper_finetune.txt &

sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b-instruct_wav2vec2_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b-instruct_wav2vec2_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b-instruct_wav2vec2_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b-instruct_wav2vec2_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b_wav2vec2_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b_wav2vec2_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b_wav2vec2_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b_wav2vec2_finetune.txt &

sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b-instruct_wav2vec2_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b-instruct_wav2vec2_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b-instruct_wav2vec2_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b-instruct_wav2vec2_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b_wav2vec2_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b_wav2vec2_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b_wav2vec2_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b_wav2vec2_finetune.txt &

sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b-instruct_wav2vec2_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b-instruct_wav2vec2_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b-instruct_wav2vec2_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b-instruct_wav2vec2_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b_wav2vec2_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b_wav2vec2_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b_wav2vec2_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b_wav2vec2_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b-instruct_wav2vec2_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b-instruct_wav2vec2_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b-instruct_wav2vec2_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b-instruct_wav2vec2_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b_wav2vec2_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b_wav2vec2_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b_wav2vec2_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b_wav2vec2_finetune.txt &

sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b-instruct_wavlm_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b-instruct_wavlm_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b-instruct_wavlm_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b-instruct_wavlm_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b_wavlm_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b_wavlm_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b_wavlm_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b_wavlm_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b-instruct_wavlm_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b-instruct_wavlm_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b-instruct_wavlm_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b-instruct_wavlm_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b_wavlm_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b_wavlm_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b_wavlm_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b_wavlm_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b-instruct_wavlm_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b-instruct_wavlm_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b-instruct_wavlm_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b-instruct_wavlm_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b_wavlm_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b_wavlm_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b_wavlm_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b_wavlm_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b-instruct_wavlm_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b-instruct_wavlm_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b-instruct_wavlm_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b-instruct_wavlm_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b_wavlm_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b_wavlm_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b_wavlm_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b_wavlm_finetune.txt &

sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b-instruct_wav2vec2-bert_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b-instruct_wav2vec2-bert_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b-instruct_wav2vec2-bert_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b-instruct_wav2vec2-bert_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b_wav2vec2-bert_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b_wav2vec2-bert_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-0.5b_wav2vec2-bert_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-0.5b_wav2vec2-bert_finetune.txt &

sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b-instruct_wav2vec2-bert_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b-instruct_wav2vec2-bert_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b-instruct_wav2vec2-bert_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b-instruct_wav2vec2-bert_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b_wav2vec2-bert_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b_wav2vec2-bert_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-1.5b_wav2vec2-bert_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-1.5b_wav2vec2-bert_finetune.txt &

sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b-instruct_wav2vec2-bert_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b-instruct_wav2vec2-bert_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b-instruct_wav2vec2-bert_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b-instruct_wav2vec2-bert_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b_wav2vec2-bert_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b_wav2vec2-bert_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-3b_wav2vec2-bert_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-3b_wav2vec2-bert_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b-instruct_wav2vec2-bert_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b-instruct_wav2vec2-bert_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b-instruct_wav2vec2-bert_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b-instruct_wav2vec2-bert_finetune.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b_wav2vec2-bert_frozen.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b_wav2vec2-bert_frozen.txt &
sh run.sh --config /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/configs/qwen2-7b_wav2vec2-bert_finetune.yaml &> AudioLLM-Deepfake-Detection/output/errors/qwen2-7b_wav2vec2-bert_finetune.txt &