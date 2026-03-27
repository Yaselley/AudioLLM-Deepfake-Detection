#!/usr/bin/env python3
"""
List all checkpoint folders matching whisperv3_unfrozen or w2v under outputs/.
Writes paths (relative to repo root) to a txt file for batch run_eval.sh.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(REPO_ROOT, "outputs")
OUT_TXT = os.path.join(REPO_ROOT, "eval_folders_whisperv3_unfrozen_and_w2v.txt")
OUT_CMDS = os.path.join(REPO_ROOT, "run_eval_batch_mlaad.sh")


def main():
    folders = []
    for parent in sorted(os.listdir(OUTPUTS)):
        pdir = os.path.join(OUTPUTS, parent)
        if not os.path.isdir(pdir):
            continue
        for name in sorted(os.listdir(pdir)):
            if "whisperv3_unfrozen" not in name and "w2v" not in name.lower():
                continue
            # skip analysis / weight folders (not checkpoint dirs)
            if "analysis" in parent.lower() or "weight" in name.lower():
                continue
            rel = os.path.join("outputs", parent, name)
            config = os.path.join(REPO_ROOT, rel, "config.json")
            if os.path.isfile(config):
                folders.append(rel)

    # deduplicate and sort
    folders = sorted(set(folders))

    with open(OUT_TXT, "w") as f:
        for p in folders:
            f.write(p + "\n")

    # Optional: write a shell script to run all with mlaad log (log under basename/eval._mlaad.txt)
    with open(OUT_CMDS, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Run from /netscratch/yelkheir/DeepFense\n")
        f.write("cd /netscratch/yelkheir/DeepFense\n\n")
        for p in folders:
            basename = os.path.basename(p)
            f.write(f"sh run_eval.sh {p} &> {basename}/eval._mlaad.txt &\n")
        f.write("\nwait\n")

    print(f"Wrote {len(folders)} paths to {OUT_TXT}")
    print(f"Wrote batch script to {OUT_CMDS}")
    for p in folders[:5]:
        print(f"  {p}")
    if len(folders) > 5:
        print(f"  ... and {len(folders) - 5} more")


if __name__ == "__main__":
    main()
