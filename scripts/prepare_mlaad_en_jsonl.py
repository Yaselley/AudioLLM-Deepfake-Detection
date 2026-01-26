#!/usr/bin/env python3
"""
Prepare JSONL file for MLAAD-En evaluation set.
"""
import os
import json

def prepare_mlaad_en_jsonl():
    flac_path = "/ds-slt/audio/yelkheir/MLAAD/MLAAD_v3/v3/MLAADv3/"
    label_file = "/netscratch/yelkheir/datasets/MLAAD/per_dialect/en.txt"
    output_file = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/data/mlaad_en_eval.jsonl"
    
    # Label encoding
    encode = {'spoof': 'fake', 'bonafide': 'real', 'fake': 'fake', 'real': 'real'}
    
    entries = []
    
    with open(label_file, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # Format: _ <path> <dialect> _ <spoof/bonafide>
            # wav_id is actually a path-like structure in parts[1]
            wav_id_path = parts[1]
            label_str = parts[4].lower()
            
            # Convert to real/fake
            if label_str in encode:
                label = encode[label_str]
            else:
                continue
            
            # The wav_id_path is a relative path from flac_path
            # Try different combinations with extensions
            wav_path = os.path.join(flac_path, f"{wav_id_path}.wav")
            if not os.path.exists(wav_path):
                wav_path = os.path.join(flac_path, f"{wav_id_path}.flac")
            if not os.path.exists(wav_path):
                # Try without adding extension (might already have it in the path)
                wav_path = os.path.join(flac_path, wav_id_path)
                if not os.path.exists(wav_path):
                    # Try with .wav extension if path doesn't have one
                    if not wav_id_path.endswith(('.wav', '.flac', '.WAV', '.FLAC')):
                        wav_path = os.path.join(flac_path, f"{wav_id_path}.wav")
            
            if os.path.exists(wav_path):
                entries.append({
                    "path": wav_path,
                    "label": label
                })
            else:
                print(f"Warning: Cannot find audio file for {wav_id_path}")
    
    # Write JSONL file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created {output_file} with {len(entries)} entries")

if __name__ == "__main__":
    prepare_mlaad_en_jsonl()
