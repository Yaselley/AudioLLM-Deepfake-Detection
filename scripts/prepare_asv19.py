import pandas as pd
import json
import os
import sys

def convert_parquet_to_jsonl(input_parquet, output_jsonl):
    print(f"Reading {input_parquet}...")
    try:
        df = pd.read_parquet(input_parquet)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    # Map labels
    label_map = {
        "bonafide": "real",
        "spoof": "fake"
    }
    
    # Check if 'label' column exists
    if 'label' not in df.columns:
        print(f"Error: 'label' column not found in {input_parquet}")
        return

    # Ensure path exists
    if 'path' not in df.columns:
        print(f"Error: 'path' column not found in {input_parquet}")
        return

    print(f"Converting {len(df)} records...")
    
    with open(output_jsonl, 'w') as f:
        for _, row in df.iterrows():
            original_label = row['label']
            if original_label in label_map:
                new_label = label_map[original_label]
                
                entry = {
                    "path": row['path'],
                    "label": new_label
                }
                f.write(json.dumps(entry) + '\n')
            else:
                print(f"Warning: Unknown label '{original_label}' found.")

    print(f"Saved to {output_jsonl}")

def main():
    base_parquet_dir = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets"
    output_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/data"
    
    os.makedirs(output_dir, exist_ok=True)

    # Train
    train_input = os.path.join(base_parquet_dir, "asvspoof19_train.parquet")
    train_output = os.path.join(output_dir, "asv19_train.jsonl")
    convert_parquet_to_jsonl(train_input, train_output)

    # Val (Validation/Dev)
    # Note: ASVspoof usually has 'dev' and 'eval'. The user asked for 'val' based on 'asvspoof19_val.parquet'
    val_input = os.path.join(base_parquet_dir, "asvspoof19_val.parquet")
    val_output = os.path.join(output_dir, "asv19_val.jsonl")
    convert_parquet_to_jsonl(val_input, val_output)

    val_input = os.path.join(base_parquet_dir, "asvspoof19_test.parquet")
    val_output = os.path.join(output_dir, "asv19_test.jsonl")
    convert_parquet_to_jsonl(val_input, val_output)

if __name__ == "__main__":
    main()
