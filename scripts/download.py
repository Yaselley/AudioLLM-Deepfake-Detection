from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="fedikallel/Qwen3.5_4B_no_lora_42_LA19",
    local_dir="/netscratch/fkallel/AudioLLM-Deepfake-Detection/output_v6/qwen3.5_4B_no_lora_42_asv19",
    token="..."
)
