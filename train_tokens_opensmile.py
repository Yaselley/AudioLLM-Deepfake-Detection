#!/usr/bin/env python3
"""Train with OpenSmile — same pipeline as train_tokens.py (enable via config)."""

from train_tokens import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenSmile training (set use_opensmile: true and <OPENSMILE> in prompt_path)"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--hf_token", default=None, help="HF token (or set HF_TOKEN)")
    args = parser.parse_args()
    main(args.config, args.hf_token)
