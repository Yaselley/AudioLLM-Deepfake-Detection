#!/usr/bin/env python3
"""
Prepare JSONL file for DF21 evaluation set (eval phase only).
"""
import os
import json

def prepare_df21_eval_jsonl():
    flac_path = "/ds/audio/DF_21/ASVspoof2021_DF_eval/flac/"
    # Use CM trial_metadata.txt to filter by phase
    label_file = "/ds/audio/DF_21/keys/DF/CM/trial_metadata.txt"
    output_file = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/data/df21_eval.jsonl"
    
    # Label encoding
    encode = {'spoof': 'fake', 'bonafide': 'real', 'fake': 'fake', 'real': 'real'}
    
    entries = []
    
    with open(label_file, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            
            # Column 7 (index 7) is the phase - we only want "eval"
            phase = parts[7] if len(parts) > 7 else None
            if phase != "eval":
                continue
            
            wav_id = parts[1]  # Column 1 is the wav_id
            
            # Column 4 (index 4) is the label
            try:
                label_str = parts[4].lower()
            except IndexError:
                continue
            
            # Convert to real/fake
            if label_str in encode:
                label = encode[label_str]
            else:
                continue
            
            # Check for both .wav and .flac extensions
            wav_path = os.path.join(flac_path, f"{wav_id}.flac")
            if not os.path.exists(wav_path):
                wav_path = os.path.join(flac_path, f"{wav_id}.wav")
            
            if os.path.exists(wav_path):
                entries.append({
                    "path": wav_path,
                    "label": label
                })
            else:
                print(f"Warning: Cannot find audio file for {wav_id}")
    
    # Write JSONL file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created {output_file} with {len(entries)} entries")

if __name__ == "__main__":
    prepare_df21_eval_jsonl()
