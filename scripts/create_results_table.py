#!/usr/bin/env python3
"""
Create a formatted CSV table from JSON results matching the specified format.
"""
import json
import csv
import numpy as np
from collections import defaultdict

def normalize_llm_name(llm_str):
    """Convert LLM model names to standardized format"""
    llm_str = llm_str.lower()
    if 'qwen2.5-0.5b' in llm_str or 'qwen2-0.5b' in llm_str:
        return 'Qwen-0.5'
    elif 'qwen2.5-3b' in llm_str or 'qwen2-3b' in llm_str:
        return 'Qwen-3'
    elif 'qwen2.5-7b' in llm_str or 'qwen2-7b' in llm_str:
        return 'Qwen-7'
    return llm_str

def normalize_encoder_name(encoder_str):
    """Convert encoder names to standardized format"""
    encoder_str = encoder_str.lower()
    if 'whisper_ner' in encoder_str or 'whisperv3_ner' in encoder_str:
        return 'Whisper-ner'
    elif 'whisper' in encoder_str:
        return 'Whisper'
    elif 'wav2vec2' in encoder_str or 'wav2vec' in encoder_str:
        return 'Wav2Vec2'
    elif 'eat' in encoder_str:
        return 'EAT'
    elif 'speechtokenizer' in encoder_str:
        # Handle different SpeechTokenizer variants
        if '1:8' in encoder_str:
            return 'SpeechTokenizer 1:8'
        elif '1:4' in encoder_str:
            return 'SpeechTokenizer 1:4'
        elif ':1' in encoder_str or '1:1' in encoder_str:
            return 'SpeechTokenizer 1:1'
        return 'SpeechTokenizer'
    return encoder_str

def normalize_dataset_name(dataset_str):
    """Convert dataset names to standardized format"""
    dataset_str = dataset_str.lower()
    if 'asv19' in dataset_str:
        return 'ASV19'
    elif 'itw' in dataset_str:
        return 'ITW'
    elif 'la21' in dataset_str:
        return 'LA21'

    elif 'mlaad' in dataset_str:
        return 'MLAAD'
    return dataset_str

def get_frozen_label(frozen_str):
    """Convert frozen status to label"""
    frozen_str = frozen_str.lower()
    if 'unfrozen' in frozen_str:
        return 'Tuned'
    elif 'frozen' in frozen_str:
        return 'Frozen'
    return frozen_str

def create_results_table(json_path, output_csv):
    """Create CSV table from JSON results"""
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Organize data: encoder -> frozen_status -> llm -> dataset -> metrics
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    target_datasets = ['ASV19', 'ITW', 'LA21', 'MLAAD']
    target_llms = ['Qwen-0.5', 'Qwen-3', 'Qwen-7']
    
    # Extract data from JSON - ONLY use pre-averaged 'avg' entries (already averaged across seeds)
    for llm_key, llm_data in data.items():
        llm_normalized = normalize_llm_name(llm_key)
        if llm_normalized not in target_llms:
            continue
        
        for encoder_key, encoder_data in llm_data.items():
            encoder_normalized = normalize_encoder_name(encoder_key)
            
            for frozen_key, frozen_data in encoder_data.items():
                # Skip if frozen_key is 'avg' itself (shouldn't happen, but safety check)
                if frozen_key == 'avg':
                    continue
                
                if isinstance(frozen_data, dict) and 'avg' in frozen_data and isinstance(frozen_data['avg'], list):
                    # Use the pre-averaged data from 'avg' entry (already averaged across seeds)
                    frozen_label = get_frozen_label(frozen_key)  # This converts 'unfrozen' to 'Tuned'
                    
                    for avg_entry in frozen_data['avg']:
                        dataset_name = normalize_dataset_name(avg_entry.get('dataset', ''))
                        if dataset_name not in target_datasets:
                            continue
                        
                        organized[encoder_normalized][frozen_label][llm_normalized][dataset_name] = {
                            'accuracy': avg_entry.get('avg_accuracy', 0),
                            'f1_score': avg_entry.get('avg_f1_score', 0),
                            'eer': avg_entry.get('avg_eer', 0) * 100  # Convert to percentage
                        }
    
    # Define encoder order and categories
    encoder_order = [
        ('Whisper', 'Speech Encoders, Semantic'),
        ('Whisper-ner', 'Prompt'),
        ('Wav2Vec2', 'Non-Semantic Tokens'),
        ('EAT', 'Non-Semantic Tokens'),
        ('SpeechTokenizer 1:8', 'Codecs'),
        ('SpeechTokenizer 1:4', 'Codecs'),
        ('SpeechTokenizer 1:1', 'Codecs'),
        ('SpeechTokenizer', 'Codecs'),
    ]
    
    # Create CSV table
    rows = []
    
    # Header row 1: LLM names spanning 12 columns each (4 datasets × 3 metrics)
    header1 = [''] + ['Qwen-0.5'] * 12 + ['Qwen-3'] * 12 + ['Qwen-7'] * 12
    rows.append(header1)
    
    # Header row 2: Dataset names, each spanning 3 columns (ACC, F1, EER)
    header2 = ['']
    for _ in range(3):  # For each LLM
        for dataset in ['ASV19', 'ITW', 'LA21', 'MLAAD', 'AVG']:
            header2.append(dataset)  # Dataset name in first column
            header2.extend(['', ''])  # Empty for F1 and EER columns
    rows.append(header2)
    
    # Header row 3: Metrics (ACC, F1, EER) repeating for each dataset
    header3 = [''] + (['ACC', 'F1', 'EER'] * 4) * 3
    rows.append(header3)
    
    # Process each encoder category
    current_category = None
    for encoder_name, category in encoder_order:
        if encoder_name not in organized:
            continue
        
        # Add category row if new
        if category != current_category:
            category_row = [category] + [''] * (12 * 3)
            rows.append(category_row)
            current_category = category
        
        encoder_data = organized[encoder_name]
        
        # Process Frozen and Tuned
        for frozen_status in ['Frozen', 'Tuned']:
            if frozen_status not in encoder_data:
                # Add empty row
                row = [f'{encoder_name}\t{frozen_status}'] + [''] * (12 * 3)
                rows.append(row)
                continue
            
            frozen_data = encoder_data[frozen_status]
            row = [f'{encoder_name}\t{frozen_status}']
            
            # For each LLM
            for llm in target_llms:
                if llm not in frozen_data:
                    # Empty cells for this LLM
                    row.extend([''] * 12)
                    continue
                
                llm_data = frozen_data[llm]
                
                # For each dataset
                for dataset in target_datasets:
                    if dataset in llm_data:
                        metrics = llm_data[dataset]
                        row.extend([
                            f"{metrics['accuracy']:.4f}",
                            f"{metrics['f1_score']:.4f}",
                            f"{metrics['eer']:.2f}"
                        ])
                    else:
                        row.extend([''] * 3)
                
                # Calculate AVG across datasets
                available_datasets = [d for d in target_datasets if d in llm_data]
                if available_datasets:
                    avg_acc = np.mean([llm_data[d]['accuracy'] for d in available_datasets])
                    avg_f1 = np.mean([llm_data[d]['f1_score'] for d in available_datasets])
                    avg_eer = np.mean([llm_data[d]['eer'] for d in available_datasets])
                    row.extend([
                        f"{avg_acc:.4f}",
                        f"{avg_f1:.4f}",
                        f"{avg_eer:.2f}"
                    ])
                else:
                    row.extend([''] * 3)
            
            rows.append(row)
    
    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)
    
    print(f"Table saved to: {output_csv}")

if __name__ == "__main__":
    json_path = '/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/all_results_organized.json'
    output_csv = '/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/results_table_prompt.csv'
    create_results_table(json_path, output_csv)
