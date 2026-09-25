from huggingface_hub import HfApi, upload_folder, create_repo

api = HfApi()

# Upload the entire folder
create_repo(
    repo_id=f"fedikallel/Qwen3.5_0.8B_lora_42_reasoningHIRSDD_stage2_HIRSDD_GRPO_stage3_HIRSDD_and_LA19_GRPO",
    repo_type="model",
    exist_ok=True,   # won't error if it already exists
    private=True,    # or False, your choice
    token="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"
)
upload_folder(
    repo_id=f"fedikallel/Qwen3.5_0.8B_lora_42_reasoningHIRSDD_stage2_HIRSDD_GRPO_stage3_HIRSDD_and_LA19_GRPO",
    folder_path=f"/netscratch/fkallel/AudioLLM-Deepfake-Detection/output_grpo_trl/lora_42_reasoning_both_stage2_output",
    path_in_repo=".",  # Upload to the root of the repo
    commit_message="Uploading model from cluster",
    repo_type="model",
    allow_patterns=[
            "evals/**",          # everything inside evals/
            "config.json",
            "train.log",
            "checkpoint_best.pt",
            "audio_encoder_best.pt",
            "proj_best.pt",
            "lora_best/**"
        ],
    token="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"
)

create_repo(
    repo_id=f"fedikallel/Qwen3.5_0.8B_lora_42_reasoningHIRSDD_stage2_lora_2_reasoningHIRSDD_GRPO",
    repo_type="model",
    exist_ok=True,   # won't error if it already exists
    private=True,    # or False, your choice
    token="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"
)
upload_folder(
    repo_id=f"fedikallel/Qwen3.5_0.8B_lora_42_reasoningHIRSDD_stage2_lora_2_reasoningHIRSDD_GRPO",
    folder_path=f"/netscratch/fkallel/AudioLLM-Deepfake-Detection/output_grpo_trl/lora_2_reasoning_stage2_output",
    path_in_repo=".",  # Upload to the root of the repo
    commit_message="Uploading model from cluster",
    repo_type="model",
    allow_patterns=[
            "evals/**",          # everything inside evals/
            "config.json",
            "train.log",
            "checkpoint_best.pt",
            "audio_encoder_best.pt",
            "proj_best.pt",
            "lora_best/**"
        ],
    token="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"
)

create_repo(
    repo_id=f"fedikallel/Qwen3.5_0.8B_lora_42_reasoningHIRSDD_stage2_lora_42_reasoningHIRSDD_GRPO",
    repo_type="model",
    exist_ok=True,   # won't error if it already exists
    private=True,    # or False, your choice
    token="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"
)
upload_folder(
    repo_id=f"fedikallel/Qwen3.5_0.8B_lora_42_reasoningHIRSDD_stage2_lora_42_reasoningHIRSDD_GRPO",
    folder_path=f"/netscratch/fkallel/AudioLLM-Deepfake-Detection/output_grpo_trl/lora_42_reasoning_stage2_output",
    path_in_repo=".",  # Upload to the root of the repo
    commit_message="Uploading model from cluster",
    repo_type="model",
    allow_patterns=[
            "evals/**",          # everything inside evals/
            "config.json",
            "train.log",
            "checkpoint_best.pt",
            "audio_encoder_best.pt",
            "proj_best.pt",
            "lora_best/**"
        ],
    token="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"
)

create_repo(
    repo_id=f"fedikallel/Qwen3.5_0.8B_lora_42_reasoningHIRSDD_stage2_lora_240_reasoningHIRSDD_GRPO",
    repo_type="model",
    exist_ok=True,   # won't error if it already exists
    private=True,    # or False, your choice
    token="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"
)
upload_folder(
    repo_id=f"fedikallel/Qwen3.5_0.8B_lora_42_reasoningHIRSDD_stage2_lora_240_reasoningHIRSDD_GRPO",
    folder_path=f"/netscratch/fkallel/AudioLLM-Deepfake-Detection/output_grpo_trl/lora_240_reasoning_stage2_output",
    path_in_repo=".",  # Upload to the root of the repo
    commit_message="Uploading model from cluster",
    repo_type="model",
    allow_patterns=[
            "evals/**",          # everything inside evals/
            "config.json",
            "train.log",
            "checkpoint_best.pt",
            "audio_encoder_best.pt",
            "proj_best.pt",
            "lora_best/**"
        ],
    token="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"
)
