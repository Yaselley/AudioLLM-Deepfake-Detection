#!/usr/bin/env python3
"""
Script to analyze all CSV results from outputs/suite and outputs/suite2.
Calculates accuracy, F1-score, and EER for each CSV file.
Organizes results by LLM model, speech feature, frozen/unfrozen status, and seed.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import det_curve, f1_score, accuracy_score
import json

# EER function from compute_eer.py
def get_eer(y_true, y_score):
    """
    A simple function to compute eer.
    More sophisticated metrics are available in this framework.
    This is a quick and easy implementation to use during validation.
    """
    fpr, fnr, thresholds = det_curve(y_true, y_score)
    index = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[index] + fnr[index]
    eer = eer/2
    return eer


def parse_directory_name(dir_name):
    """
    Parse directory name to extract LLM model, speech feature, frozen status, and seed.
    
    Example: 'qwen2.5-0.5b-instruct_whisperv3_ner_frozen_seed42'
    Returns: {
        'llm_model': 'qwen2.5-0.5b-instruct',
        'speech_feature': 'whisperv3_ner',
        'frozen': 'frozen',
        'seed': '42'
    }
    """
    # Pattern: {llm_model}_{speech_feature}_{frozen/unfrozen}_seed{seed}
    # Handle variations in naming
    
    # Extract seed (last part after 'seed')
    seed_match = re.search(r'seed(\d+)', dir_name)
    seed = seed_match.group(1) if seed_match else 'unknown'
    
    # Extract frozen/unfrozen
    if '_frozen_' in dir_name or dir_name.endswith('_frozen'):
        frozen = 'frozen'
        # Remove frozen part for further parsing
        remaining = re.sub(r'_frozen(_|$)', '_', dir_name)
    elif '_unfrozen_' in dir_name or dir_name.endswith('_unfrozen'):
        frozen = 'unfrozen'
        # Remove unfrozen part for further parsing
        remaining = re.sub(r'_unfrozen(_|$)', '_', dir_name)
    else:
        frozen = 'unknown'
        remaining = dir_name
    
    # Remove seed part
    remaining = re.sub(r'_seed\d+$', '', remaining)
    remaining = remaining.rstrip('_')
    
    # Split by underscore to get parts
    parts = remaining.split('_')
    
    # LLM model is typically the first part(s) - usually contains 'qwen', 'llama', etc.
    # Speech feature is the middle part(s)
    # Find where LLM model ends (usually after 'instruct' or similar)
    llm_parts = []
    speech_parts = []
    
    found_instruct = False
    for i, part in enumerate(parts):
        if 'instruct' in part.lower():
            llm_parts.append(part)
            found_instruct = True
        elif found_instruct:
            speech_parts.append(part)
        else:
            llm_parts.append(part)
    
    # If we didn't find 'instruct', try to find common LLM patterns
    if not found_instruct:
        for i, part in enumerate(parts):
            if any(x in part.lower() for x in ['qwen', 'llama', 'gpt', 'bert']):
                llm_parts.append(part)
                found_instruct = True
            elif found_instruct:
                speech_parts.append(part)
            else:
                llm_parts.append(part)
    
    llm_model = '_'.join(llm_parts) if llm_parts else 'unknown'
    speech_feature = '_'.join(speech_parts) if speech_parts else 'unknown'
    
    return {
        'llm_model': llm_model,
        'speech_feature': speech_feature,
        'frozen': frozen,
        'seed': seed
    }


def calculate_metrics(csv_path):
    """
    Calculate accuracy, F1-score, and EER from a CSV file.
    
    CSV format expected:
    - ground_truth: 'Real' or 'Fake'
    - prediction: 'Real' or 'Fake'
    - score_real: probability score for real
    - score_fake: probability score for fake
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Convert ground_truth and prediction to binary (1 for Real, 0 for Fake)
        y_true = (df['ground_truth'].str.lower() == 'real').astype(int).values
        y_pred = (df['prediction'].str.lower() == 'real').astype(int).values
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate F1-score (handles imbalanced data)
        f1 = f1_score(y_true, y_pred, average='macro')  # Macro F1 for imbalanced data
        
        y_scores_fakes = df['score_fake'].values.astype(float)
        y_scores_real = df['score_real'].values.astype(float)
        
        y_scores = y_scores_real - y_scores_fakes
        
        eer = get_eer(y_true, y_scores)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'eer': eer,
            'num_samples': len(df)
        }
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


def find_all_csv_files(base_dirs):
    """
    Find all CSV files ending with '_best.csv' in the given directories.
    Returns list of (csv_path, relative_path) tuples.
    """
    csv_files = []
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Warning: Directory {base_dir} does not exist")
            continue
        
        # Find all CSV files ending with '_best.csv'
        for csv_file in base_path.rglob('*_best_42.csv'):
            # Get relative path from base_dir
            rel_path = csv_file.relative_to(base_path)
            csv_files.append((csv_file, rel_path))
    
    return csv_files


def organize_results(all_results):
    """
    Organize results by LLM model, speech feature, frozen status, and seed.
    Returns nested dictionary structure.
    """
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for result in all_results:
        if result is None:
            continue
        
        llm = result['llm_model']
        speech = result['speech_feature']
        frozen = result['frozen']
        seed = result['seed']
        dataset = result['dataset']
        
        organized[llm][speech][frozen][seed].append({
            'dataset': dataset,
            'accuracy': result['accuracy'],
            'f1_score': result['f1_score'],
            'eer': result['eer'],
            'num_samples': result['num_samples'],
            'csv_path': result['csv_path']
        })
    
    return organized


def extract_dataset_name(csv_filename):
    """
    Extract dataset name from CSV filename.
    Examples: 'asv19_test_best.csv' -> 'asv19_test'
              'itw_best.csv' -> 'itw'
              'la21_best.csv' -> 'la21'
    """
    # Remove '_best.csv' suffix
    name = csv_filename.replace('_best.csv', '')
    return name


def main():
    # Base directories to search
    base_dirs = [
        "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/suite_prompt",
    ]
    
    print("Finding all CSV files...")
    csv_files = find_all_csv_files(base_dirs)
    print(f"Found {len(csv_files)} CSV files")
    
    all_results = []
    
    print("\nProcessing CSV files...")
    for csv_path, rel_path in csv_files:
        # Extract directory name (parent of CSV file)
        dir_name = rel_path.parent.name
        csv_filename = rel_path.name
        dataset = extract_dataset_name(csv_filename)
        
        # Parse directory name
        parsed = parse_directory_name(dir_name)
        
        # Calculate metrics
        metrics = calculate_metrics(csv_path)
        
        if metrics is not None:
            result = {
                **parsed,
                'dataset': dataset,
                'csv_path': str(csv_path),
                **metrics
            }
            all_results.append(result)
            print(f"Processed: {dir_name}/{csv_filename}")
        else:
            print(f"Failed to process: {dir_name}/{csv_filename}")
    
    print(f"\nSuccessfully processed {len(all_results)} CSV files")
    
    # Organize results
    organized = organize_results(all_results)
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Create a flat summary for easier viewing
    summary_data = []
    
    for llm in sorted(organized.keys()):
        for speech in sorted(organized[llm].keys()):
            for frozen in sorted(organized[llm][speech].keys()):
                for seed in sorted(organized[llm][speech][frozen].keys()):
                    # Skip 'avg' entries that we added
                    if seed == 'avg':
                        continue
                    
                    datasets_info = organized[llm][speech][frozen][seed]
                    
                    if not isinstance(datasets_info, list):
                        continue
                    
                    for dataset_info in datasets_info:
                        summary_data.append({
                            'llm_model': llm,
                            'speech_feature': speech,
                            'frozen': frozen,
                            'seed': seed,
                            'dataset': dataset_info['dataset'],
                            'accuracy': dataset_info['accuracy'],
                            'f1_score': dataset_info['f1_score'],
                            'eer': dataset_info['eer'],
                            'num_samples': dataset_info['num_samples']
                        })
    
    # Create DataFrame for easier viewing
    df_summary = pd.DataFrame(summary_data)
    
    # Print summary table
    print("\nDetailed Results:")
    print(df_summary.to_string(index=False))
    
    # Save to CSV
    output_csv = '/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/all_results_summary_prompt.csv'
    df_summary.to_csv(output_csv, index=False)
    print(f"\nSummary saved to: {output_csv}")
    
    # Add averages to organized structure before saving to JSON
    # For each dataset: average across seeds for this configuration (llm + encoder + frozen/unfrozen)
    for llm in organized.keys():
        for speech in organized[llm].keys():
            for frozen in organized[llm][speech].keys():
                # Collect all datasets and their values across seeds
                dataset_results = defaultdict(list)  # dataset_name -> list of results from different seeds
                
                for seed in list(organized[llm][speech][frozen].keys()):
                    datasets_info = organized[llm][speech][frozen][seed]
                    
                    if not datasets_info or not isinstance(datasets_info, list):
                        continue
                    
                    # Collect results per dataset
                    for dataset_info in datasets_info:
                        dataset_name = dataset_info['dataset']
                        dataset_results[dataset_name].append({
                            'seed': str(seed),
                            'accuracy': dataset_info['accuracy'],
                            'f1_score': dataset_info['f1_score'],
                            'eer': dataset_info['eer'],
                            'num_samples': dataset_info['num_samples']
                        })
                
                # Calculate averages across seeds for each dataset
                avg_by_dataset = []
                
                for dataset_name in sorted(dataset_results.keys()):
                    results = dataset_results[dataset_name]
                    
                    if not results:
                        continue
                    
                    accuracies = [r['accuracy'] for r in results]
                    f1_scores = [r['f1_score'] for r in results]
                    eers = [r['eer'] for r in results]
                    num_samples = [r['num_samples'] for r in results]
                    seeds_with_data = [r['seed'] for r in results]
                    
                    avg_by_dataset.append({
                        'dataset': dataset_name,
                        'avg_accuracy': float(np.mean(accuracies)),
                        'std_accuracy': float(np.std(accuracies)),
                        'avg_f1_score': float(np.mean(f1_scores)),
                        'std_f1_score': float(np.std(f1_scores)),
                        'avg_eer': float(np.mean(eers)),
                        'std_eer': float(np.std(eers)),
                        'num_seeds': len(results),
                        'seeds': sorted(seeds_with_data),
                        'num_samples': int(np.mean(num_samples))  # Average sample count
                    })
                
                # Add "avg" key at frozen level (averaged across seeds per dataset)
                if avg_by_dataset:
                    organized[llm][speech][frozen]['avg'] = avg_by_dataset
    
    # Convert defaultdict to regular dict for JSON serialization
    def convert_to_dict(d):
        if isinstance(d, defaultdict):
            return {k: convert_to_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_to_dict(item) if isinstance(item, (dict, defaultdict)) else item for item in d]
        elif isinstance(d, dict):
            return {k: convert_to_dict(v) for k, v in d.items()}
        else:
            return d
    
    organized_dict = convert_to_dict(organized)
    output_json = '/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/all_results_organized.json'
    with open(output_json, 'w') as f:
        json.dump(organized_dict, f, indent=2)
    print(f"Organized results saved to: {output_json}")
    
    # Average across datasets per seed for each configuration (llm + encoder + frozen/unfrozen)
    print("\n" + "="*80)
    print("AVERAGE ACROSS DATASETS PER CONFIGURATION (by seed)")
    print("="*80)
    
    # Group by llm_model, speech_feature, frozen, and seed, then average metrics
    config_avg_data = []
    
    for llm in sorted(organized.keys()):
        for speech in sorted(organized[llm].keys()):
            for frozen in sorted(organized[llm][speech].keys()):
                for seed in sorted(organized[llm][speech][frozen].keys()):
                    # Skip 'avg' entries that we added
                    if seed == 'avg':
                        continue
                    
                    datasets_info = organized[llm][speech][frozen][seed]
                    
                    if not datasets_info or not isinstance(datasets_info, list):
                        continue
                    
                    # Check if this is the expected structure (has 'accuracy' key)
                    if not datasets_info or not all(isinstance(d, dict) and 'accuracy' in d for d in datasets_info):
                        continue
                    
                    # Calculate averages across all datasets for this configuration
                    accuracies = [d['accuracy'] for d in datasets_info]
                    f1_scores = [d['f1_score'] for d in datasets_info]
                    eers = [d['eer'] for d in datasets_info]
                    num_samples = [d['num_samples'] for d in datasets_info]
                    
                    # Also collect dataset names for reference
                    dataset_names = [d['dataset'] for d in datasets_info]
                    
                    config_avg_data.append({
                        'llm_model': llm,
                        'speech_feature': speech,
                        'frozen': frozen,
                        'seed': seed,
                        'num_datasets': len(datasets_info),
                        'datasets': ', '.join(sorted(dataset_names)),
                        'avg_accuracy': np.mean(accuracies),
                        'std_accuracy': np.std(accuracies),
                        'avg_f1_score': np.mean(f1_scores),
                        'std_f1_score': np.std(f1_scores),
                        'avg_eer': np.mean(eers),
                        'std_eer': np.std(eers),
                        'total_samples': sum(num_samples)
                    })
    
    # Create DataFrame for averaged results
    df_config_avg = pd.DataFrame(config_avg_data)
    
    # Print averaged results table
    print("\nAveraged Results Across Datasets (per configuration and seed):")
    print(df_config_avg.to_string(index=False))
    
    # Save averaged results to CSV
    output_avg_csv = '/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/all_results_averaged_prompt.csv'
    df_config_avg.to_csv(output_avg_csv, index=False)
    print(f"\nAveraged results saved to: {output_avg_csv}")
    
    # Average across seeds for each configuration (llm + encoder + frozen/unfrozen)
    print("\n" + "="*80)
    print("AVERAGE ACROSS SEEDS PER CONFIGURATION")
    print("="*80)
    
    # Group by llm_model, speech_feature, and frozen, then average across seeds
    config_seed_avg_data = []
    
    for llm in sorted(organized.keys()):
        for speech in sorted(organized[llm].keys()):
            for frozen in sorted(organized[llm][speech].keys()):
                seeds_data = []
                all_seeds = []
                
                for seed in sorted(organized[llm][speech][frozen].keys()):
                    # Skip 'avg' entries that we added
                    if seed == 'avg':
                        continue
                    
                    datasets_info = organized[llm][speech][frozen][seed]
                    if not datasets_info or not isinstance(datasets_info, list):
                        continue
                    
                    all_seeds.append(seed)
                    # Calculate averages across datasets for this seed
                    accuracies = [d['accuracy'] for d in datasets_info]
                    f1_scores = [d['f1_score'] for d in datasets_info]
                    eers = [d['eer'] for d in datasets_info]
                    
                    seeds_data.append({
                        'avg_accuracy': np.mean(accuracies),
                        'avg_f1_score': np.mean(f1_scores),
                        'avg_eer': np.mean(eers),
                        'num_datasets': len(datasets_info)
                    })
                
                if not seeds_data:
                    continue
                
                # Average across seeds
                all_accuracies = [s['avg_accuracy'] for s in seeds_data]
                all_f1_scores = [s['avg_f1_score'] for s in seeds_data]
                all_eers = [s['avg_eer'] for s in seeds_data]
                all_num_datasets = [s['num_datasets'] for s in seeds_data]
                
                config_seed_avg_data.append({
                    'llm_model': llm,
                    'speech_feature': speech,
                    'frozen': frozen,
                    'num_seeds': len(all_seeds),
                    'seeds': ', '.join(sorted(all_seeds)),
                    'avg_accuracy': np.mean(all_accuracies),
                    'std_accuracy': np.std(all_accuracies),
                    'avg_f1_score': np.mean(all_f1_scores),
                    'std_f1_score': np.std(all_f1_scores),
                    'avg_eer': np.mean(all_eers),
                    'std_eer': np.std(all_eers),
                    'avg_num_datasets': np.mean(all_num_datasets)
                })
    
    # Create DataFrame for seed-averaged results
    df_config_seed_avg = pd.DataFrame(config_seed_avg_data)
    
    # Print seed-averaged results table
    print("\nAveraged Results Across Seeds (per configuration):")
    print(df_config_seed_avg.to_string(index=False))
    
    # Save seed-averaged results to CSV
    output_seed_avg_csv = '/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/all_results_averaged_across_seeds_prompt.csv'
    df_config_seed_avg.to_csv(output_seed_avg_csv, index=False)
    print(f"\nSeed-averaged results saved to: {output_seed_avg_csv}")
    
    # Print statistics by dimension
    print("\n" + "="*80)
    print("STATISTICS BY DIMENSION")
    print("="*80)
    
    # By LLM Model
    print("\nAverage Metrics by LLM Model:")
    llm_stats = df_summary.groupby('llm_model').agg({
        'accuracy': 'mean',
        'f1_score': 'mean',
        'eer': 'mean'
    }).round(4)
    print(llm_stats)
    
    # By Speech Feature
    print("\nAverage Metrics by Speech Feature:")
    speech_stats = df_summary.groupby('speech_feature').agg({
        'accuracy': 'mean',
        'f1_score': 'mean',
        'eer': 'mean'
    }).round(4)
    print(speech_stats)
    
    # By Frozen Status
    print("\nAverage Metrics by Frozen Status:")
    frozen_stats = df_summary.groupby('frozen').agg({
        'accuracy': 'mean',
        'f1_score': 'mean',
        'eer': 'mean'
    }).round(4)
    print(frozen_stats)
    
    # By Dataset
    print("\nAverage Metrics by Dataset:")
    dataset_stats = df_summary.groupby('dataset').agg({
        'accuracy': 'mean',
        'f1_score': 'mean',
        'eer': 'mean'
    }).round(4)
    print(dataset_stats)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
