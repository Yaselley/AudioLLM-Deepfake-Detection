#!/usr/bin/env python3
"""
Create a formatted CSV table from SpeechTokenizer JSON results.
Similar format to create_results_table.py
"""
import json
import csv
import numpy as np
from collections import defaultdict

def normalize_llm_name(llm_size):
    """Convert LLM size to standardized format"""
    if llm_size == '0.5':
        return 'Qwen-0.5'
    elif llm_size == '3':
        return 'Qwen-3'
    elif llm_size == '7':
        return 'Qwen-7'
    return f'Qwen-{llm_size}'

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

def create_speechtokenizer_table(json_path, output_csv):
    """Create CSV table from SpeechTokenizer JSON results"""
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Organize data: mode -> layers -> lora_status -> llm -> dataset -> metrics
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    
    target_datasets = ['ASV19', 'ITW', 'LA21', 'MLAAD']
    target_llms = ['Qwen-0.5', 'Qwen-3', 'Qwen-7']
    
    # Extract data from JSON
    speechtokenizer_data = data.get('SpeechTokenizer', {})
    
    for mode in ['discrete', 'continuous']:
        if mode not in speechtokenizer_data:
            continue
        
        mode_data = speechtokenizer_data[mode]
        
        for layers in ['1', '4', '8']:
            if layers not in mode_data:
                continue
            
            layers_data = mode_data[layers]
            
            # Process LLM_Lora and LLM_Frozen
            for lora_status_key, lora_status_label in [('LLM_Lora', 'Lora'), ('LLM_Frozen', 'Frozen')]:
                if lora_status_key not in layers_data:
                    continue
                
                lora_data = layers_data[lora_status_key]
                if lora_data is None:
                    continue
                
                # Get detailed results
                detailed_results = lora_data.get('detailed_results', [])
                
                # Organize by LLM and dataset
                for result in detailed_results:
                    llm_size = result.get('llm_size', '')
                    dataset = normalize_dataset_name(result.get('dataset', ''))
                    
                    if dataset not in target_datasets:
                        continue
                    
                    llm_normalized = normalize_llm_name(llm_size)
                    if llm_normalized not in target_llms:
                        continue
                    
                    organized[mode][layers][lora_status_label][llm_normalized][dataset] = {
                        'accuracy': result.get('accuracy', 0),
                        'f1_score': result.get('f1_fake', 0),
                        'eer': result.get('eer', 0) * 100  # Convert to percentage
                    }
    
    # Create CSV table
    rows = []
    
    # Header row 1: LLM names spanning 15 columns each (5 datasets × 3 metrics)
    header1 = [''] + ['Qwen-0.5'] * 15 + ['Qwen-3'] * 15 + ['Qwen-7'] * 15
    rows.append(header1)
    
    # Header row 2: Dataset names, each spanning 3 columns (ACC, F1, EER)
    header2 = ['']
    for _ in range(3):  # For each LLM
        for dataset in ['ASV19', 'ITW', 'LA21', 'MLAAD', 'AVG']:
            header2.append(dataset)  # Dataset name in first column
            header2.extend(['', ''])  # Empty for F1 and EER columns
    rows.append(header2)
    
    # Header row 3: Metrics (ACC, F1, EER) repeating for each dataset
    header3 = [''] + (['ACC', 'F1', 'EER'] * 5) * 3
    rows.append(header3)
    
    # Process each mode and layer configuration
    # Order: discrete:1, discrete:4, discrete:8, continuous:1, continuous:4, continuous:8
    configurations = [
        ('discrete', '1', 'Lora'),
        ('discrete', '4', 'Lora'),
        ('discrete', '8', 'Lora'),
        ('continuous', '1', 'Lora'),
        ('continuous', '4', 'Lora'),
        ('continuous', '8', 'Lora'),
        ('continuous', '1', 'Frozen'),
        ('continuous', '4', 'Frozen'),
        ('continuous', '8', 'Frozen'),
    ]
    
    for mode, layers, lora_status in configurations:
        row_label = f"SpeechTokenizer\t{mode}:{layers}\t{lora_status}"
        row = [row_label]
        
        # For each LLM
        for llm in target_llms:
            if mode not in organized or layers not in organized[mode] or lora_status not in organized[mode][layers]:
                # Empty cells for this LLM
                row.extend([''] * 15)
                continue
            
            llm_data = organized[mode][layers][lora_status].get(llm, {})
            
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
    json_path = '/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/speechtokenizer_results.json'
    output_csv = '/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/speechtokenizer_table.csv'
    create_speechtokenizer_table(json_path, output_csv)
