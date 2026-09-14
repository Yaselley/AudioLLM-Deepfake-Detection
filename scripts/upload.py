from huggingface_hub import HfApi, upload_folder, create_repo

api = HfApi()

# Upload the entire folder
# for size in ["0.8B", "4B"]:
#     for seed in ["2", "42", "240"]:
#         create_repo(
#             repo_id=f"fedikallel/Qwen3.5_{size}_lora_{seed}_LA19_stage2_binaryHIRSDD",
#             repo_type="model",
#             exist_ok=True,   # won't error if it already exists
#             private=False,    # or False, your choice
#             token="..."
#         )
#         upload_folder(
#             repo_id=f"fedikallel/Qwen3.5_{size}_lora_{seed}_LA19_stage2_binaryHIRSDD",
#             folder_path=f"/netscratch/fkallel/AudioLLM-Deepfake-Detection/output_v6/qwen3.5_{size}_lora_{seed}_myhir_asvspoof19",
#             path_in_repo=".",  # Upload to the root of the repo
#             commit_message="Uploading model from cluster",
#             repo_type="model",
#             ignore_patterns=["*stage2_reasoning_output*","*stage2_output*", "*stage2_explanation_output*", "*real_stage2_output*"],
#             token="..."
#         )

for size in ["0.8B", "4B"]:
    for seed in ["2", "42", "240"]:
        create_repo(
            repo_id=f"fedikallel/Qwen3.5_{size}_no_lora_{seed}_LA19_stage2_binaryHIRSDD",
            repo_type="model",
            exist_ok=True,   # won't error if it already exists
            private=False,    # or False, your choice
            token="..."
        )
        upload_folder(
            repo_id=f"fedikallel/Qwen3.5_{size}_no_lora_{seed}_LA19_stage2_binaryHIRSDD",
            folder_path=f"/netscratch/fkallel/AudioLLM-Deepfake-Detection/output_v6/qwen3.5_{size}_no_lora_{seed}_asv19/real_stage2_output",
            path_in_repo=".",  # Upload to the root of the repo
            commit_message="Uploading model from cluster",
            repo_type="model",
            ignore_patterns=["*stage2_reasoning_output*","*stage2_output*", "*stage2_explanation_output*", "*real_stage2_output*"],
            token="..."
        )
