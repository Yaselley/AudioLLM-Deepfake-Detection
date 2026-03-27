#!/usr/bin/env python3
"""
Script to analyze DAC results from outputs/suite_tokenizers.
Organizes results by discrete/continuous mode, codebooks (mapped to layers), and LoRA/Frozen status.
Outputs as JSON.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict


def parse_dac_directory(dir_name):
    """
    Parse directory name to extract configuration.
    
    Examples:
    - 'qwen2.5-0.5b-instruct_dac_1_lora_discrete' -> discrete, 1 codebook (1 layer), lora
    - 'qwen2.5-3b-instruct_dac_6_lora_continuous' -> continuous, 6 codebooks (4 layers), lora
    - 'qwen2.5-7b-instruct_dac_12_nolora_continuous' -> continuous, 12 codebooks (8 layers), frozen
    
    Mapping: 1 codebook -> 1 layer, 6 codebooks -> 4 layers, 12 codebooks -> 8 layers
    
    Returns:
        dict with keys: 'llm_size', 'codebooks', 'layers', 'mode', 'lora_status'
    """
    # Extract codebooks (number after dac_)
    codebooks_match = re.search(r'dac_(\d+)_', dir_name)
    codebooks = int(codebooks_match.group(1)) if codebooks_match else None
    
    # Map codebooks to layers
    codebook_to_layer = {1: 1, 6: 4, 12: 8}
    layers = codebook_to_layer.get(codebooks, None)
    
    # Determine mode: discrete or continuous
    if '_continuous' in dir_name:
        mode = 'continuous'
    elif '_discrete' in dir_name:
        mode = 'discrete'
    else:
        mode = 'unknown'
    
    # Determine LoRA status
    if '_nolora_' in dir_name or dir_name.endswith('_nolora_continuous'):
        lora_status = 'frozen'  # No LoRA = Frozen LLM
    elif '_lora' in dir_name:
        lora_status = 'lora'  # Has LoRA = LLM Lora
    else:
        lora_status = 'unknown'
    
    # Extract LLM size (0.5b, 3b, 7b)
    llm_match = re.search(r'qwen2\.5-(\d+\.?\d*)b-instruct', dir_name)
    llm_size = llm_match.group(1) if llm_match else 'unknown'
    
    return {
        'llm_size': llm_size,
        'codebooks': codebooks,
        'layers': layers,
        'mode': mode,
        'lora_status': lora_status,
        'dir_name': dir_name
    }


def load_metrics_json(json_path):
    """
    Load metrics from JSON file.
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def extract_dataset_name(json_filename):
    """
    Extract dataset name from JSON filename.
    Example: 'asv19_test_best_42_metrics_best_42.json' -> 'asv19_test'
    """
    name = json_filename.replace('_best_42_metrics_best_42.json', '')
    return name


def find_dac_results(base_dir):
    """
    Find all DAC result directories and their JSON metrics files.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Warning: Directory {base_dir} does not exist")
        return []
    
    results = []
    
    # Find all directories containing 'dac'
    for dir_path in base_path.iterdir():
        if not dir_path.is_dir():
            continue
        
        dir_name = dir_path.name
        if 'dac' not in dir_name.lower():
            continue
        
        # Parse directory name
        config = parse_dac_directory(dir_name)
        
        if config['codebooks'] is None or config['layers'] is None:
            continue
        
        # Find all metrics JSON files in this directory
        json_files = list(dir_path.glob('*_best_42_metrics_best_42.json'))
        
        for json_file in json_files:
            dataset = extract_dataset_name(json_file.name)
            metrics = load_metrics_json(json_file)
            
            if metrics is not None:
                results.append({
                    **config,
                    'dataset': dataset,
                    'metrics': metrics,
                    'json_path': str(json_file)
                })
    
    return results


def organize_results(all_results):
    """
    Organize results by the structure requested:
    - discrete:1, discrete:4, discrete:8 (all with LLM Lora)
    - continuous:1, continuous:4, continuous:8 (with LLM Lora)
    - continuous:1, continuous:4, continuous:8 (with LLM Frozen)
    """
    organized = {
        'discrete': {
            '1': {'lora': []},
            '4': {'lora': []},
            '8': {'lora': []}
        },
        'continuous': {
            '1': {'lora': [], 'frozen': []},
            '4': {'lora': [], 'frozen': []},
            '8': {'lora': [], 'frozen': []}
        }
    }
    
    for result in all_results:
        mode = result['mode']
        layers = str(result['layers'])
        lora_status = result['lora_status']
        
        # Skip if layers not in [1, 4, 8]
        if layers not in ['1', '4', '8']:
            continue
        
        # Skip if mode is discrete but lora_status is not lora
        if mode == 'discrete' and lora_status != 'lora':
            continue
        
        # Skip if mode is continuous but lora_status is not lora or frozen
        if mode == 'continuous' and lora_status not in ['lora', 'frozen']:
            continue
        
        # Add to organized structure
        if mode in organized and layers in organized[mode]:
            if lora_status in organized[mode][layers]:
                organized[mode][layers][lora_status].append({
                    'llm_size': result['llm_size'],
                    'codebooks': result['codebooks'],
                    'dataset': result['dataset'],
                    'accuracy': result['metrics'].get('accuracy'),
                    'f1_fake': result['metrics'].get('f1_fake'),
                    'eer': result['metrics'].get('eer'),
                    'total_samples': result['metrics'].get('total_samples'),
                    'json_path': result['json_path']
                })
    
    return organized


def calculate_averages(organized):
    """
    Calculate averages across LLM sizes and datasets for each configuration.
    """
    averaged = {}
    
    for mode in ['discrete', 'continuous']:
        averaged[mode] = {}
        for layers in ['1', '4', '8']:
            averaged[mode][layers] = {}
            
            for lora_status in organized[mode][layers].keys():
                results = organized[mode][layers][lora_status]
                
                if not results:
                    averaged[mode][layers][lora_status] = None
                    continue
                
                # Collect all metrics
                accuracies = [r['accuracy'] for r in results if r['accuracy'] is not None]
                f1_scores = [r['f1_fake'] for r in results if r['f1_fake'] is not None]
                eers = [r['eer'] for r in results if r['eer'] is not None]
                
                # Get unique datasets
                datasets = sorted(set(r['dataset'] for r in results))
                
                # Get unique LLM sizes
                llm_sizes = sorted(set(r['llm_size'] for r in results))
                
                averaged[mode][layers][lora_status] = {
                    'avg_accuracy': sum(accuracies) / len(accuracies) if accuracies else None,
                    'avg_f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else None,
                    'avg_eer': sum(eers) / len(eers) if eers else None,
                    'num_results': len(results),
                    'datasets': datasets,
                    'llm_sizes': llm_sizes,
                    'detailed_results': results
                }
    
    return averaged


def main():
    base_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/suite_tokenizers"
    
    print("Finding DAC results...")
    all_results = find_dac_results(base_dir)
    print(f"Found {len(all_results)} result files")
    
    print("\nOrganizing results...")
    organized = organize_results(all_results)
    
    print("\nCalculating averages...")
    averaged = calculate_averages(organized)
    
    # Create final output structure
    output = {
        'DAC': {
            'discrete': {
                '1': {
                    'LLM_Lora': averaged['discrete']['1']['lora']
                },
                '4': {
                    'LLM_Lora': averaged['discrete']['4']['lora']
                },
                '8': {
                    'LLM_Lora': averaged['discrete']['8']['lora']
                }
            },
            'continuous': {
                '1': {
                    'LLM_Lora': averaged['continuous']['1']['lora'],
                    'LLM_Frozen': averaged['continuous']['1']['frozen']
                },
                '4': {
                    'LLM_Lora': averaged['continuous']['4']['lora'],
                    'LLM_Frozen': averaged['continuous']['4']['frozen']
                },
                '8': {
                    'LLM_Lora': averaged['continuous']['8']['lora'],
                    'LLM_Frozen': averaged['continuous']['8']['frozen']
                }
            }
        }
    }
    
    # Save to JSON
    output_json = '/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/dac_results.json'
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_json}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for mode in ['discrete', 'continuous']:
        print(f"\n{mode.upper()}:")
        for layers in ['1', '4', '8']:
            print(f"  Layers {layers}:")
            for lora_status in organized[mode][layers].keys():
                count = len(organized[mode][layers][lora_status])
                avg_data = averaged[mode][layers][lora_status]
                if avg_data:
                    print(f"    {lora_status.upper()}: {count} results")
                    if avg_data['avg_accuracy']:
                        print(f"      Avg Accuracy: {avg_data['avg_accuracy']:.4f}")
                        print(f"      Avg F1 Score: {avg_data['avg_f1_score']:.4f}")
                        print(f"      Avg EER: {avg_data['avg_eer']:.4f}")
                        print(f"      Datasets: {', '.join(avg_data['datasets'])}")
                        print(f"      LLM Sizes: {', '.join(avg_data['llm_sizes'])}")
                else:
                    print(f"    {lora_status.upper()}: No results")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
