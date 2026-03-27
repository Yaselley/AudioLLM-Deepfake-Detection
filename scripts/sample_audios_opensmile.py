"""
Sample 10 real + 10 fake from asv19_train and from mlaad_en;
copy audios into a folder with names fake_asv19_0, real_asv19_0, ..., fake_mlaad_0, ...;
extract OpenSMILE features (precision 6) and write one txt file.
"""
import json
import shutil
import sys
from pathlib import Path

# Project root and paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ASV19_JSONL = DATA_DIR / "asv19_train.jsonl"
MLAAD_JSONL = DATA_DIR / "mlaad_en.jsonl"
OUT_AUDIO_DIR = REPO_ROOT / "data" / "sample_audios_opensmile"
OUT_FEATURES_TXT = OUT_AUDIO_DIR / "opensmile_features.txt"

N_PER_CLASS = 10
PRECISION = 6


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def sample_real_fake(entries, n=10):
    real = [e for e in entries if e.get("label") == "real"]
    fake = [e for e in entries if e.get("label") == "fake"]
    return {
        "real": real[:n],
        "fake": fake[:n],
    }


def get_ext(path_str):
    return Path(path_str).suffix or ".wav"


def main():
    # 1) Load and sample
    print("Loading asv19_train.jsonl...")
    asv19 = load_jsonl(ASV19_JSONL)
    asv19_sampled = sample_real_fake(asv19, N_PER_CLASS)

    print("Loading mlaad_en.jsonl...")
    mlaad = load_jsonl(MLAAD_JSONL)
    mlaad_sampled = sample_real_fake(mlaad, N_PER_CLASS)

    # 2) Create output folder and copy files
    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    # List of (local_name, source_path, label) for feature extraction
    to_process = []

    for i in range(N_PER_CLASS):
        # asv19 fake
        src = asv19_sampled["fake"][i]["path"]
        ext = get_ext(src)
        name = f"fake_asv19_{i}{ext}"
        dst = OUT_AUDIO_DIR / name
        shutil.copy2(src, dst)
        to_process.append((name, str(dst), "fake"))

        # asv19 real
        src = asv19_sampled["real"][i]["path"]
        ext = get_ext(src)
        name = f"real_asv19_{i}{ext}"
        dst = OUT_AUDIO_DIR / name
        shutil.copy2(src, dst)
        to_process.append((name, str(dst), "real"))

    for i in range(N_PER_CLASS):
        # mlaad fake
        src = mlaad_sampled["fake"][i]["path"]
        ext = get_ext(src)
        name = f"fake_mlaad_{i}{ext}"
        dst = OUT_AUDIO_DIR / name
        shutil.copy2(src, dst)
        to_process.append((name, str(dst), "fake"))

        # mlaad real
        src = mlaad_sampled["real"][i]["path"]
        ext = get_ext(src)
        name = f"real_mlaad_{i}{ext}"
        dst = OUT_AUDIO_DIR / name
        shutil.copy2(src, dst)
        to_process.append((name, str(dst), "real"))

    print(f"Copied {len(to_process)} audio files to {OUT_AUDIO_DIR}")

    # 3) OpenSMILE features (precision 6) into one txt
    # Import here so opensmile is only required when actually extracting
    sys.path.insert(0, str(REPO_ROOT / "models_clean" / "speech_encoders"))
    from opensmile_features import OpenSMILETextExtractor

    print("Initializing OpenSMILE extractor...")
    extractor = OpenSMILETextExtractor()

    with open(OUT_FEATURES_TXT, "w") as out_f:
        for i, (local_name, audio_path, label) in enumerate(to_process, 1):
            print(f"[{i}/{len(to_process)}] {local_name}")
            try:
                feature_text = extractor.process_file(audio_path, precision=PRECISION)
                out_f.write(f"=== {local_name} ===\n")
                out_f.write(f"label: {label}\n")
                out_f.write(f"{feature_text}\n\n")
            except Exception as e:
                out_f.write(f"=== {local_name} ===\n")
                out_f.write(f"label: {label}\n")
                out_f.write(f"ERROR: {e}\n\n")
                print(f"  Error: {e}")

    print(f"OpenSMILE features (precision {PRECISION}) written to {OUT_FEATURES_TXT}")


if __name__ == "__main__":
    main()
