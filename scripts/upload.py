from huggingface_hub import HfApi, upload_folder, create_repo

api = HfApi()

# Upload the entire folder
upload_folder(
    repo_id=f"fedikallel/LLM-as-a-judge-outputs",
    folder_path=f"/netscratch/fkallel/AudioLLM-Deepfake-Detection/",
    path_in_repo=".",  # Upload to the root of the repo
    commit_message="Uploading model from cluster",
    repo_type="dataset",
    allow_patterns=[
        "**/judge_google_gemma-4-31B-it_outputs.jsonl",
        "**/judge_scores_google_gemma-4-31B-it.jsonl"
        ],
    token="..."
)

