"""
Extract OpenSMILE features from all audio files in asv19_train.jsonl
and create distribution plots comparing fake vs real.
"""
import json
import os
import pandas as pd
import numpy as np
import opensmile
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
JSONL_PATH = PROJECT_ROOT / "data" / "asv19_train.jsonl"
OUTPUT_CSV = SCRIPT_DIR / "asv19_train_opensmile_features.csv"
PLOTS_DIR = SCRIPT_DIR / "feature_distributions"
NUM_FEATURES_TO_PLOT = None  # None = plot all features
MAX_SAMPLES = None  # Limit number of samples for testing (None = process all)

def extract_features():
    """Extract OpenSMILE features from all audio files."""
    print("Initializing OpenSMILE...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    print(f"Reading {JSONL_PATH}...")
    with open(str(JSONL_PATH), 'r') as f:
        all_entries = [json.loads(line) for line in f]
    
    print(f"Found {len(all_entries)} total audio files")
    
    # Separate real and fake entries
    real_entries = [e for e in all_entries if e['label'] == 'real']
    fake_entries = [e for e in all_entries if e['label'] == 'fake']
    
    print(f"  Real: {len(real_entries)}")
    print(f"  Fake: {len(fake_entries)}")
    
    # Limit samples for testing if specified
    if MAX_SAMPLES and MAX_SAMPLES < len(all_entries):
        # Randomly sample from each class proportionally
        n_real = int(MAX_SAMPLES * len(real_entries) / len(all_entries))
        n_fake = MAX_SAMPLES - n_real
        
        # Ensure we don't sample more than available
        n_real = min(n_real, len(real_entries))
        n_fake = min(n_fake, len(fake_entries))
        
        # If we still need more samples, adjust
        if n_real + n_fake < MAX_SAMPLES:
            if n_real < len(real_entries):
                n_real = min(MAX_SAMPLES - n_fake, len(real_entries))
            elif n_fake < len(fake_entries):
                n_fake = min(MAX_SAMPLES - n_real, len(fake_entries))
        
        print(f"\nRandomly sampling {n_real} real and {n_fake} fake samples (total: {n_real + n_fake})")
        random.seed(42)  # For reproducibility
        random.shuffle(real_entries)
        random.shuffle(fake_entries)
        entries = real_entries[:n_real] + fake_entries[:n_fake]
        random.shuffle(entries)  # Shuffle combined list
    else:
        entries = all_entries
    
    print(f"Processing {len(entries)} audio files")
    
    # Extract features
    all_features = []
    failed_files = []
    
    for entry in tqdm(entries, desc="Extracting features"):
        audio_path = entry['path']
        label = entry['label']
        
        if not os.path.exists(audio_path):
            failed_files.append(audio_path)
            continue
        
        try:
            features = smile.process_file(audio_path)
            # Convert to dict and add metadata
            feat_dict = features.iloc[0].to_dict()
            feat_dict['label'] = label
            feat_dict['file_path'] = audio_path
            feat_dict['file_name'] = os.path.basename(audio_path)
            all_features.append(feat_dict)
        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            failed_files.append(audio_path)
    
    if failed_files:
        print(f"\nWarning: {len(failed_files)} files failed to process")
        with open("failed_files.txt", 'w') as f:
            f.write('\n'.join(failed_files))
    
    # Convert to DataFrame
    print("\nCreating DataFrame...")
    df = pd.DataFrame(all_features)
    
    # Save to CSV
    print(f"Saving to {OUTPUT_CSV}...")
    df.to_csv(str(OUTPUT_CSV), index=False)
    print(f"Saved {len(df)} samples with {len(df.columns)-3} features")
    
    # Print summary
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    
    return df

def create_distribution_plots(df):
    """Create distribution plots for each feature comparing fake vs real."""
    print("\nCreating distribution plots...")
    
    # Create output directory
    os.makedirs(str(PLOTS_DIR), exist_ok=True)
    
    # Get feature columns (exclude metadata columns)
    feature_cols = [col for col in df.columns if col not in ['label', 'file_path', 'file_name']]
    
    print(f"Found {len(feature_cols)} features")
    if NUM_FEATURES_TO_PLOT:
        print(f"Will create plots for first {min(NUM_FEATURES_TO_PLOT, len(feature_cols))} features")
    else:
        print(f"Will create plots for all {len(feature_cols)} features")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # Create plots for each feature
    features_to_plot = feature_cols[:NUM_FEATURES_TO_PLOT] if NUM_FEATURES_TO_PLOT else feature_cols
    for i, feature in enumerate(tqdm(features_to_plot, desc="Creating plots")):
        try:
            # Remove NaN values for this feature
            plot_data = df[['label', feature]].dropna()
            
            if len(plot_data) == 0:
                continue
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Histogram/KDE overlay
            ax1 = axes[0]
            for label in ['real', 'fake']:
                data = plot_data[plot_data['label'] == label][feature]
                if len(data) > 0:
                    sns.histplot(data=data, label=label, alpha=0.6, kde=True, ax=ax1)
            ax1.set_xlabel(feature)
            ax1.set_ylabel('Density')
            ax1.set_title(f'Distribution: {feature}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Box plot
            ax2 = axes[1]
            sns.boxplot(data=plot_data, x='label', y=feature, ax=ax2)
            ax2.set_title(f'Box Plot: {feature}')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            # Sanitize feature name for filename
            safe_name = feature.replace('/', '_').replace('\\', '_').replace(' ', '_')
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c in ('_', '-', '.'))[:100]
            plt.savefig(str(PLOTS_DIR / f"{safe_name}.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"\nError creating plot for {feature}: {e}")
            continue
    
    print(f"\nPlots saved to {PLOTS_DIR}/")
    
    # Create a summary plot with multiple features
    print("\nCreating summary plot...")
    n_features = min(12, len(feature_cols))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_cols[:n_features]):
        try:
            plot_data = df[['label', feature]].dropna()
            if len(plot_data) > 0:
                sns.boxplot(data=plot_data, x='label', y=feature, ax=axes[i])
                axes[i].set_title(feature[:30], fontsize=9)
                axes[i].tick_params(labelsize=8)
        except:
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "summary_12_features.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Summary plot saved!")

def main():
    """Main function."""
    print("=" * 60)
    print("OpenSMILE Feature Extraction and Visualization")
    print("=" * 60)
    
    # Extract features
    df = extract_features()
    
    # Create distribution plots
    create_distribution_plots(df)
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Features CSV: {OUTPUT_CSV}")
    print(f"Distribution plots: {PLOTS_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
