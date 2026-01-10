import os
import json
import argparse
def main(args):
    input_txt = args.input_txt
    audio_dir = args.audio_dir
    output_jsonl = args.output_jsonl
    
    print(f"Reading from {input_txt}")
    print(f"Audio directory: {audio_dir}")
    print(f"Output: {output_jsonl}")
    
    entries = []
    
    with open(input_txt, 'r') as f:
        lines = f.readlines()
        
    count = 0
    for line in lines:
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} lines...", end='\r')
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        # Format: ID ID _ _ LABEL
        file_id = parts[0]
        label_str = parts[4] # "spoof" or "bonafide"
        
        # Map label
        if label_str == "spoof":
            label = "fake"
        elif label_str == "bonafide":
            label = "real"
        else:
            print(f"Unknown label: {label_str}")
            continue
            
        # Construct path
        file_name = f"{file_id}.wav"
        full_path = os.path.join(audio_dir, file_name)
        
        # Verify existence (optional, can be slow for network drives)
        if not os.path.exists(full_path):
            # Try padding? Some datasets do that. But list showed 10016.wav
            # Let's assume no padding based on ls output.
            # print(f"File not found: {full_path}")
            # continue
            pass # Trusting the file exists or letting the dataloader handle it
            
        entry = {
            "path": full_path,
            "label": label
        }
        entries.append(entry)
        
    print(f"Found {len(entries)} entries.")
    
    with open(output_jsonl, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_txt", type=str, required=True, help="Path to eval.final.txt")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to output JSONL")
    
    args = parser.parse_args()
    main(args)
