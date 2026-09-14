from huggingface_hub import HfApi, upload_folder, create_repo

api = HfApi()
for size in ["0.8B"]:
    for seed in ["2", "42", "240"]:
        create_repo(
            repo_id=f"fedikallel/Qwen3.5_{size}_binaryHIRSDD_and_LA19_stage2_lora_{seed}_reasoningHIRSDD",
            repo_type="model",
            exist_ok=True,   # won't error if it already exists
            private=False,    # or False, your choice
            token="..."
        )
        upload_folder(
            repo_id=f"fedikallel/Qwen3.5_{size}_binaryHIRSDD_and_LA19_stage2_lora_{seed}_reasoningHIRSDD",
            folder_path=f"/netscratch/fkallel/AudioLLM-Deepfake-Detection/output_v5/qwen3.5_0.8B_no_lora_2_myhir_asvspoof19/lora_{seed}_reasoning_stage2_output",
            path_in_repo=".",  # Upload to the root of the repo
            commit_message="Uploading model from cluster",
            repo_type="model",
            ignore_patterns=["*epoch_ckpts*","*stage2_reasoning_output*","*stage2_output*", "*stage2_explanation_output*", "*real_stage2_output*"],
            token="..."
        )
