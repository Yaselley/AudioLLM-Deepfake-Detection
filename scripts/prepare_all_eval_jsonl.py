#!/usr/bin/env python3
"""
Master script to prepare all evaluation JSONL files.
"""
import subprocess
import sys
import os

def main():
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        "prepare_mlaad_en_jsonl.py",
        "prepare_la21_eval_jsonl.py",
        "prepare_df21_eval_jsonl.py"
    ]
    
    for script in scripts:
        script_path = os.path.join(scripts_dir, script)
        print(f"\n{'='*60}")
        print(f"Running {script}...")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                capture_output=False
            )
            print(f"✓ Successfully completed {script}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running {script}: {e}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("All JSONL files prepared successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
