#!/usr/bin/env python3
"""Upload results/ to Hugging Face Hub as 01Yassine/AudioLLM-Deepfake-Detection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_REPO = "01Yassine/AudioLLM-Deepfake-Detection"


def get_token(explicit: str | None) -> str:
    if explicit:
        return explicit
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.is_file():
        return token_path.read_text().strip()
    raise SystemExit(
        "No Hugging Face token found. Set HF_TOKEN or run: hf auth login"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", action="store_true", help="Make repo public")
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset"),
        help="Hub repo type (default: model — contains checkpoints + evals)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only verify auth and repo creation",
    )
    parser.add_argument("--token", default=None, help="HF token (else HF_TOKEN env)")
    args = parser.parse_args()

    if not RESULTS.is_dir():
        raise SystemExit(f"Missing results dir: {RESULTS}")

    token = get_token(args.token)
    private = not args.public

    try:
        from huggingface_hub import HfApi, upload_large_folder
    except ImportError:
        raise SystemExit("Install: pip install -U huggingface_hub")

    api = HfApi(token=token)
    who = api.whoami()
    print(f"Authenticated as: {who.get('name', who)}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=private,
        exist_ok=True,
    )
    print(f"Repo ready: https://huggingface.co/{args.repo_type}s/{args.repo_id}")
    print(f"Private: {private}")

    if args.dry_run:
        print("Dry run — skipping upload.")
        return

    # Model card at repo root (HF displays README.md)
    readme = RESULTS / "README.md"
    if readme.is_file():
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message="Add results README",
        )
        print("Uploaded README.md")

    print(f"Starting large-folder upload from {RESULTS} (~283 GB)...")
    print("This is resumable; re-run the script if interrupted.")
    upload_large_folder(
        folder_path=str(RESULTS),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=private,
        num_workers=4,
    )
    print(f"Upload complete: https://huggingface.co/{args.repo_type}s/{args.repo_id}")


if __name__ == "__main__":
    main()
