#!/usr/bin/env python3
"""
Analyze all evaluation metrics and create comparison tables
"""
import os
import json
import pandas as pd
from pathlib import Path
import re

def parse_directory_name(dir_name):
    """Extract model info from directory name"""
    # Pattern: qwen{size}-{size}b-instruct_{encoder}_{freeze_status}
    # Examples:
    # - qwen0.5-0.5b-instruct_wav2vec_finetune
    # - qwen1.5-1.5b-instruct_whisper_frozen
    
    parts = dir_name.split('_')
    
    # Extract model size
    size_match = re.search(r'qwen(\d+\.?\d*)-(\d+\.?\d*)b', dir_name)
    if size_match:
        model_size = size_match.group(2) + 'B'
    else:
        model_size = 'unknown'
    
    # Extract encoder type
    encoder = 'unknown'
    if 'wav2vec' in dir_name:
        encoder = 'wav2vec2'
    elif 'whisper' in dir_name:
        encoder = 'whisper'
    
    # Extract freeze status
    freeze_status = 'unknown'
    if 'frozen' in dir_name:
        freeze_status = 'frozen'
    elif 'finetune' in dir_name:
        freeze_status = 'finetune'
    
    return {
        'model_size': model_size,
        'encoder': encoder,
        'freeze_status': freeze_status,
        'dir_name': dir_name
    }

def parse_metrics_file(filepath):
    """Parse metrics JSON file and extract info from filename"""
    filename = os.path.basename(filepath)
    
    # Pattern: {dataset}_{epoch}_metrics_{epoch}.json
    # Examples: asv19_test_1_metrics_1.json, itw_5_metrics_5.json
    
    match = re.match(r'(.+?)_(\d+)_metrics_\d+\.json', filename)
    if match:
        dataset = match.group(1)
        epoch = int(match.group(2))
    else:
        # Try alternative pattern
        match = re.match(r'(.+?)_metrics_(\d+)\.json', filename)
        if match:
            dataset = match.group(1)
            epoch = int(match.group(2))
        else:
            dataset = 'unknown'
            epoch = 0
    
    # Load metrics
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    return {
        'dataset': dataset,
        'epoch': epoch,
        'metrics': metrics
    }

def collect_all_metrics(output_dir):
    """Collect all metrics from all subdirectories"""
    all_data = []
    
    output_path = Path(output_dir)
    
    # Find all metrics JSON files
    for metrics_file in output_path.rglob("*_metrics_*.json"):
        # Get directory name
        dir_name = metrics_file.parent.name
        
        # Parse directory info
        dir_info = parse_directory_name(dir_name)
        
        # Parse metrics file
        file_info = parse_metrics_file(str(metrics_file))
        
        # Combine all info
        row = {
            **dir_info,
            **file_info,
            'filepath': str(metrics_file)
        }
        
        all_data.append(row)
    
    return all_data

def create_comparison_table(data):
    """Create comparison tables"""
    df = pd.DataFrame(data)
    
    # Flatten metrics
    metrics_cols = ['accuracy', 'precision_fake', 'recall_fake', 'f1_fake', 
                    'hallucination_rate', 'tp', 'tn', 'fp', 'fn', 'total_samples']
    
    for col in metrics_cols:
        df[col] = df['metrics'].apply(lambda x: x.get(col, None))
    
    # Drop the metrics dict column
    df = df.drop(columns=['metrics', 'filepath'])
    
    return df

def get_best_epoch_per_config(df):
    """Get best epoch (highest F1) for each configuration"""
    best_rows = []
    
    for (size, encoder, freeze, dataset), group in df.groupby(['model_size', 'encoder', 'freeze_status', 'dataset']):
        # Find row with highest F1 score
        best_idx = group['f1_fake'].idxmax()
        best_rows.append(group.loc[best_idx])
    
    return pd.DataFrame(best_rows)

def analyze_epoch_variation(df):
    """Analyze variation in metrics across epochs for each configuration"""
    variation_data = []
    
    for (size, encoder, freeze, dataset), group in df.groupby(['model_size', 'encoder', 'freeze_status', 'dataset']):
        if len(group) < 2:  # Need at least 2 epochs to calculate variation
            continue
        
        # Calculate statistics
        f1_scores = group['f1_fake'].dropna()
        acc_scores = group['accuracy'].dropna()
        
        if len(f1_scores) < 2:
            continue
        
        f1_mean = f1_scores.mean()
        f1_std = f1_scores.std()
        f1_min = f1_scores.min()
        f1_max = f1_scores.max()
        f1_range = f1_max - f1_min
        f1_cv = (f1_std / f1_mean) * 100 if f1_mean > 0 else 0  # Coefficient of variation
        
        acc_mean = acc_scores.mean()
        acc_std = acc_scores.std()
        acc_min = acc_scores.min()
        acc_max = acc_scores.max()
        acc_range = acc_max - acc_min
        
        # Get epochs
        epochs = sorted(group['epoch'].tolist())
        
        variation_data.append({
            'model_size': size,
            'encoder': encoder,
            'freeze_status': freeze,
            'dataset': dataset,
            'num_epochs': len(group),
            'epochs': ','.join(map(str, epochs)),
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'f1_min': f1_min,
            'f1_max': f1_max,
            'f1_range': f1_range,
            'f1_cv': f1_cv,  # Coefficient of variation (%)
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'acc_min': acc_min,
            'acc_max': acc_max,
            'acc_range': acc_range,
            'best_epoch': group.loc[group['f1_fake'].idxmax(), 'epoch'],
            'worst_epoch': group.loc[group['f1_fake'].idxmin(), 'epoch'],
        })
    
    return pd.DataFrame(variation_data)

def create_summary_tables(df_best):
    """Create summary comparison tables"""
    
    # Table 1: Encoder Comparison (Wav2Vec2 vs Whisper)
    encoder_comparison = []
    for (size, freeze, dataset), group in df_best.groupby(['model_size', 'freeze_status', 'dataset']):
        wav2vec = group[group['encoder'] == 'wav2vec2']
        whisper = group[group['encoder'] == 'whisper']
        
        if not wav2vec.empty and not whisper.empty:
            wav2vec_row = wav2vec.iloc[0]
            whisper_row = whisper.iloc[0]
            
            encoder_comparison.append({
                'Model Size': size,
                'Freeze Status': freeze,
                'Dataset': dataset,
                'Wav2Vec2 F1': wav2vec_row['f1_fake'],
                'Whisper F1': whisper_row['f1_fake'],
                'Wav2Vec2 Acc': wav2vec_row['accuracy'],
                'Whisper Acc': whisper_row['accuracy'],
                'Wav2Vec2 Hallucination': wav2vec_row['hallucination_rate'],
                'Whisper Hallucination': whisper_row['hallucination_rate'],
                'Best Encoder': 'Wav2Vec2' if wav2vec_row['f1_fake'] > whisper_row['f1_fake'] else 'Whisper',
                'F1 Difference': abs(wav2vec_row['f1_fake'] - whisper_row['f1_fake'])
            })
    
    # Table 2: Freeze Status Comparison
    freeze_comparison = []
    for (size, encoder, dataset), group in df_best.groupby(['model_size', 'encoder', 'dataset']):
        frozen = group[group['freeze_status'] == 'frozen']
        finetune = group[group['freeze_status'] == 'finetune']
        
        if not frozen.empty and not finetune.empty:
            frozen_row = frozen.iloc[0]
            finetune_row = finetune.iloc[0]
            
            freeze_comparison.append({
                'Model Size': size,
                'Encoder': encoder,
                'Dataset': dataset,
                'Frozen F1': frozen_row['f1_fake'],
                'Finetune F1': finetune_row['f1_fake'],
                'Frozen Acc': frozen_row['accuracy'],
                'Finetune Acc': finetune_row['accuracy'],
                'Frozen Hallucination': frozen_row['hallucination_rate'],
                'Finetune Hallucination': finetune_row['hallucination_rate'],
                'Best Freeze Status': 'Frozen' if frozen_row['f1_fake'] > finetune_row['f1_fake'] else 'Finetune',
                'F1 Difference': abs(frozen_row['f1_fake'] - finetune_row['f1_fake'])
            })
    
    # Table 3: Model Size Comparison
    size_comparison = []
    for (encoder, freeze, dataset), group in df_best.groupby(['encoder', 'freeze_status', 'dataset']):
        sizes = ['0.5B', '1.5B', '3B', '7B']
        size_data = {}
        for size in sizes:
            size_group = group[group['model_size'] == size]
            if not size_group.empty:
                size_data[size] = size_group.iloc[0]
        
        if len(size_data) >= 2:
            row = {
                'Encoder': encoder,
                'Freeze Status': freeze,
                'Dataset': dataset,
            }
            for size in sizes:
                if size in size_data:
                    row[f'{size} F1'] = size_data[size]['f1_fake']
                    row[f'{size} Acc'] = size_data[size]['accuracy']
                else:
                    row[f'{size} F1'] = None
                    row[f'{size} Acc'] = None
            
            # Find best size
            best_size = max([s for s in sizes if s in size_data], 
                          key=lambda s: size_data[s]['f1_fake'])
            row['Best Size'] = best_size
            size_comparison.append(row)
    
    return pd.DataFrame(encoder_comparison), pd.DataFrame(freeze_comparison), pd.DataFrame(size_comparison)

def generate_insights(df_best, encoder_df, freeze_df, size_df, df_variation):
    """Generate key insights"""
    insights = []
    
    # Insight 1: Overall best configuration
    best_overall = df_best.loc[df_best['f1_fake'].idxmax()]
    insights.append(f"🏆 Best Overall Configuration:")
    insights.append(f"   Model: {best_overall['model_size']}, Encoder: {best_overall['encoder']}, "
                   f"Freeze: {best_overall['freeze_status']}, Dataset: {best_overall['dataset']}")
    insights.append(f"   F1 Score: {best_overall['f1_fake']:.4f}, Accuracy: {best_overall['accuracy']:.4f}")
    insights.append("")
    
    # Insight: Epoch Variation
    if not df_variation.empty:
        insights.append(f"📊 Epoch-to-Epoch Variation Analysis:")
        avg_f1_range = df_variation['f1_range'].mean()
        max_f1_range = df_variation['f1_range'].max()
        avg_f1_cv = df_variation['f1_cv'].mean()
        max_f1_cv = df_variation['f1_cv'].max()
        
        insights.append(f"   Average F1 Range (max-min): {avg_f1_range:.4f}")
        insights.append(f"   Maximum F1 Range: {max_f1_range:.4f}")
        insights.append(f"   Average Coefficient of Variation: {avg_f1_cv:.2f}%")
        insights.append(f"   Maximum Coefficient of Variation: {max_f1_cv:.2f}%")
        insights.append("")
        
        # Find most unstable configurations
        most_unstable = df_variation.nlargest(5, 'f1_range')[['model_size', 'encoder', 'freeze_status', 'dataset', 'f1_range', 'f1_cv']]
        insights.append(f"   ⚠️  Most Unstable Configurations (Top 5 by F1 Range):")
        for idx, row in most_unstable.iterrows():
            insights.append(f"      {row['model_size']} {row['encoder']} {row['freeze_status']} on {row['dataset']}: "
                          f"Range={row['f1_range']:.4f}, CV={row['f1_cv']:.2f}%")
        insights.append("")
        
        # Find most stable configurations
        most_stable = df_variation.nsmallest(5, 'f1_range')[['model_size', 'encoder', 'freeze_status', 'dataset', 'f1_range', 'f1_cv']]
        insights.append(f"   ✅ Most Stable Configurations (Top 5 by F1 Range):")
        for idx, row in most_stable.iterrows():
            insights.append(f"      {row['model_size']} {row['encoder']} {row['freeze_status']} on {row['dataset']}: "
                          f"Range={row['f1_range']:.4f}, CV={row['f1_cv']:.2f}%")
        insights.append("")
    
    # Insight 2: Encoder comparison
    if not encoder_df.empty:
        wav2vec_wins = (encoder_df['Best Encoder'] == 'Wav2Vec2').sum()
        whisper_wins = (encoder_df['Best Encoder'] == 'Whisper').sum()
        insights.append(f"📊 Encoder Comparison:")
        insights.append(f"   Wav2Vec2 wins: {wav2vec_wins} configurations")
        insights.append(f"   Whisper wins: {whisper_wins} configurations")
        avg_diff = encoder_df['F1 Difference'].mean()
        insights.append(f"   Average F1 difference: {avg_diff:.4f}")
        insights.append("")
    
    # Insight 3: Freeze status comparison
    if not freeze_df.empty:
        frozen_wins = (freeze_df['Best Freeze Status'] == 'Frozen').sum()
        finetune_wins = (freeze_df['Best Freeze Status'] == 'Finetune').sum()
        insights.append(f"🔒 Freeze Status Comparison:")
        insights.append(f"   Frozen wins: {frozen_wins} configurations")
        insights.append(f"   Finetune wins: {finetune_wins} configurations")
        avg_diff = freeze_df['F1 Difference'].mean()
        insights.append(f"   Average F1 difference: {avg_diff:.4f}")
        insights.append("")
    
    # Insight 4: Model size trends
    if not size_df.empty:
        insights.append(f"📈 Model Size Trends:")
        for size in ['0.5B', '1.5B', '3B', '7B']:
            col = f'{size} F1'
            if col in size_df.columns:
                avg_f1 = size_df[col].mean()
                insights.append(f"   {size}: Average F1 = {avg_f1:.4f}")
        insights.append("")
    
    return "\n".join(insights)

def main():
    output_dir = "/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/output3"
    
    print("Collecting all metrics...")
    data = collect_all_metrics(output_dir)
    print(f"Found {len(data)} metrics files")
    
    print("Creating comparison tables...")
    df = create_comparison_table(data)
    
    print("Finding best epochs per configuration...")
    df_best = get_best_epoch_per_config(df)
    
    print("Analyzing epoch-to-epoch variation...")
    df_variation = analyze_epoch_variation(df)
    
    print("Creating summary tables...")
    encoder_df, freeze_df, size_df = create_summary_tables(df_best)
    
    # Save all tables as CSV files
    output_dir_path = Path(output_dir)
    df_best.to_csv(output_dir_path / "metrics_best_epochs.csv", index=False)
    encoder_df.to_csv(output_dir_path / "metrics_encoder_comparison.csv", index=False)
    freeze_df.to_csv(output_dir_path / "metrics_freeze_comparison.csv", index=False)
    size_df.to_csv(output_dir_path / "metrics_size_comparison.csv", index=False)
    df_variation.to_csv(output_dir_path / "metrics_epoch_variation.csv", index=False)
    df.to_csv(output_dir_path / "metrics_all_data.csv", index=False)
    
    print(f"Saved analysis tables to {output_dir_path}/")
    print("  - metrics_best_epochs.csv")
    print("  - metrics_encoder_comparison.csv")
    print("  - metrics_freeze_comparison.csv")
    print("  - metrics_size_comparison.csv")
    print("  - metrics_epoch_variation.csv")
    print("  - metrics_all_data.csv")
    
    # Generate and print insights
    insights = generate_insights(df_best, encoder_df, freeze_df, size_df, df_variation)
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print(insights)
    
    # Save insights
    insights_file = os.path.join(output_dir, "metrics_insights.txt")
    with open(insights_file, 'w') as f:
        f.write(insights)
    print(f"\nInsights saved to {insights_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTotal configurations: {len(df_best)}")
    print(f"Average F1 Score: {df_best['f1_fake'].mean():.4f}")
    print(f"Average Accuracy: {df_best['accuracy'].mean():.4f}")
    print(f"Average Hallucination Rate: {df_best['hallucination_rate'].mean():.4f}")

if __name__ == "__main__":
    main()
