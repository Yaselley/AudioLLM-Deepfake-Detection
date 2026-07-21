"""
OpenSMILE Feature Text Generator
Extracts eGeMAPSv02 features and generates textual descriptions.
"""
import json
import opensmile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors that generate textual descriptions."""
    
    def __init__(self):
        self.feature_names = []
        self.num_features = 0
    
    @abstractmethod
    def extract_features(self, audio_path: str) -> Dict[str, float]:
        """Extract features from audio file."""
        pass
    
    @abstractmethod
    def features_to_text(self, features: Dict[str, float]) -> str:
        """Convert features to textual description."""
        pass


class OpenSMILETextExtractor(BaseFeatureExtractor):
    """
    Extract OpenSMILE eGeMAPSv02 features and generate textual descriptions.
    
    eGeMAPSv02 provides 88 acoustic features including:
    - Fundamental frequency (F0) statistics
    - Formant frequencies (F1, F2, F3)
    - Spectral features (alpha ratio, hammarberg index, spectral slopes)
    - Energy and loudness
    - MFCCs and their statistics
    - Jitter and shimmer (voice quality)
    """
    
    # Human-readable feature name mappings
    FEATURE_DESCRIPTIONS = {
        'F0semitoneFrom27.5Hz_sma3nz_amean': 'Average Fundamental Frequency in Semitones from 27.5 Hz',
        'F0semitoneFrom27.5Hz_sma3nz_stddevNorm': 'Standard Deviation of F0 (normalized)',
        'F0semitoneFrom27.5Hz_sma3nz_percentile20.0': 'F0 20th Percentile',
        'F0semitoneFrom27.5Hz_sma3nz_percentile50.0': 'F0 50th Percentile (Median)',
        'F0semitoneFrom27.5Hz_sma3nz_percentile80.0': 'F0 80th Percentile',
        'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2': 'F0 Percentile Range (20-80)',
        'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope': 'Mean Rising Slope of F0',
        'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope': 'Standard Deviation of Rising Slope',
        'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope': 'Mean Falling Slope of F0',
        'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope': 'Standard Deviation of Falling Slope',
        'loudness_sma3_amean': 'Average Loudness',
        'loudness_sma3_stddevNorm': 'Standard Deviation of Loudness (normalized)',
        'loudness_sma3_percentile20.0': 'Loudness 20th Percentile',
        'loudness_sma3_percentile50.0': 'Loudness 50th Percentile (Median)',
        'loudness_sma3_percentile80.0': 'Loudness 80th Percentile',
        'loudness_sma3_pctlrange0-2': 'Loudness Percentile Range (20-80)',
        'loudness_sma3_meanRisingSlope': 'Mean Rising Slope of Loudness',
        'loudness_sma3_stddevRisingSlope': 'Standard Deviation of Loudness Rising Slope',
        'loudness_sma3_meanFallingSlope': 'Mean Falling Slope of Loudness',
        'loudness_sma3_stddevFallingSlope': 'Standard Deviation of Loudness Falling Slope',
        'spectralFlux_sma3_amean': 'Average Spectral Flux',
        'spectralFlux_sma3_stddevNorm': 'Standard Deviation of Spectral Flux (normalized)',
        'jitterLocal_sma3nz_amean': 'Average Local Jitter',
        'jitterLocal_sma3nz_stddevNorm': 'Standard Deviation of Local Jitter (normalized)',
        'shimmerLocaldB_sma3nz_amean': 'Average Local Shimmer (dB)',
        'shimmerLocaldB_sma3nz_stddevNorm': 'Standard Deviation of Local Shimmer (normalized)',
        'HNRdBACF_sma3nz_amean': 'Average Harmonics-to-Noise Ratio (dB)',
        'HNRdBACF_sma3nz_stddevNorm': 'Standard Deviation of HNR (normalized)',
        'logRelF0-H1-H2_sma3nz_amean': 'Average Log Relative F0-H1-H2',
        'logRelF0-H1-H2_sma3nz_stddevNorm': 'Standard Deviation of Log Relative F0-H1-H2 (normalized)',
        'logRelF0-H1-A3_sma3nz_amean': 'Average Log Relative F0-H1-A3',
        'logRelF0-H1-A3_sma3nz_stddevNorm': 'Standard Deviation of Log Relative F0-H1-A3 (normalized)',
        'F1frequency_sma3nz_amean': 'Average First Formant Frequency (F1)',
        'F1frequency_sma3nz_stddevNorm': 'Standard Deviation of F1 (normalized)',
        'F1bandwidth_sma3nz_amean': 'Average First Formant Bandwidth',
        'F1bandwidth_sma3nz_stddevNorm': 'Standard Deviation of F1 Bandwidth (normalized)',
        'F1amplitudeLogRelF0_sma3nz_amean': 'Average F1 Amplitude Log Relative to F0',
        'F1amplitudeLogRelF0_sma3nz_stddevNorm': 'Standard Deviation of F1 Amplitude Log Rel F0 (normalized)',
        'F2frequency_sma3nz_amean': 'Average Second Formant Frequency (F2)',
        'F2frequency_sma3nz_stddevNorm': 'Standard Deviation of F2 (normalized)',
        'F2bandwidth_sma3nz_amean': 'Average Second Formant Bandwidth',
        'F2bandwidth_sma3nz_stddevNorm': 'Standard Deviation of F2 Bandwidth (normalized)',
        'F2amplitudeLogRelF0_sma3nz_amean': 'Average F2 Amplitude Log Relative to F0',
        'F2amplitudeLogRelF0_sma3nz_stddevNorm': 'Standard Deviation of F2 Amplitude Log Rel F0 (normalized)',
        'F3frequency_sma3nz_amean': 'Average Third Formant Frequency (F3)',
        'F3frequency_sma3nz_stddevNorm': 'Standard Deviation of F3 (normalized)',
        'F3bandwidth_sma3nz_amean': 'Average Third Formant Bandwidth',
        'F3bandwidth_sma3nz_stddevNorm': 'Standard Deviation of F3 Bandwidth (normalized)',
        'F3amplitudeLogRelF0_sma3nz_amean': 'Average F3 Amplitude Log Relative to F0',
        'F3amplitudeLogRelF0_sma3nz_stddevNorm': 'Standard Deviation of F3 Amplitude Log Rel F0 (normalized)',
        'alphaRatioV_sma3nz_amean': 'Average Alpha Ratio (Voiced)',
        'alphaRatioV_sma3nz_stddevNorm': 'Standard Deviation of Alpha Ratio (normalized)',
        'hammarbergIndexV_sma3nz_amean': 'Average Hammarberg Index (Voiced)',
        'hammarbergIndexV_sma3nz_stddevNorm': 'Standard Deviation of Hammarberg Index (normalized)',
        'slopeV0-500_sma3nz_amean': 'Average Spectral Slope 0-500 Hz (Voiced)',
        'slopeV0-500_sma3nz_stddevNorm': 'Standard Deviation of Slope 0-500 Hz (normalized)',
        'slopeV500-1500_sma3nz_amean': 'Average Spectral Slope 500-1500 Hz (Voiced)',
        'slopeV500-1500_sma3nz_stddevNorm': 'Standard Deviation of Slope 500-1500 Hz (normalized)',
        'alphaRatioUV_sma3nz_amean': 'Average Alpha Ratio (Unvoiced)',
        'hammarbergIndexUV_sma3nz_amean': 'Average Hammarberg Index (Unvoiced)',
        'slopeUV0-500_sma3nz_amean': 'Average Spectral Slope 0-500 Hz (Unvoiced)',
        'slopeUV500-1500_sma3nz_amean': 'Average Spectral Slope 500-1500 Hz (Unvoiced)',
        'mfcc1_sma3_amean': 'Average MFCC 1',
        'mfcc1_sma3_stddevNorm': 'Standard Deviation of MFCC 1 (normalized)',
        'mfcc2_sma3_amean': 'Average MFCC 2',
        'mfcc2_sma3_stddevNorm': 'Standard Deviation of MFCC 2 (normalized)',
        'mfcc3_sma3_amean': 'Average MFCC 3',
        'mfcc3_sma3_stddevNorm': 'Standard Deviation of MFCC 3 (normalized)',
        'mfcc4_sma3_amean': 'Average MFCC 4',
        'mfcc4_sma3_stddevNorm': 'Standard Deviation of MFCC 4 (normalized)',
        # Voiced segment features
        'spectralFluxV_sma3nz_amean': 'Average Spectral Flux (Voiced)',
        'spectralFluxV_sma3nz_stddevNorm': 'Standard Deviation of Spectral Flux (Voiced, normalized)',
        'mfcc1V_sma3nz_amean': 'Average MFCC 1 (Voiced)',
        'mfcc1V_sma3nz_stddevNorm': 'Standard Deviation of MFCC 1 (Voiced, normalized)',
        'mfcc2V_sma3nz_amean': 'Average MFCC 2 (Voiced)',
        'mfcc2V_sma3nz_stddevNorm': 'Standard Deviation of MFCC 2 (Voiced, normalized)',
        'mfcc3V_sma3nz_amean': 'Average MFCC 3 (Voiced)',
        'mfcc3V_sma3nz_stddevNorm': 'Standard Deviation of MFCC 3 (Voiced, normalized)',
        'mfcc4V_sma3nz_amean': 'Average MFCC 4 (Voiced)',
        'mfcc4V_sma3nz_stddevNorm': 'Standard Deviation of MFCC 4 (Voiced, normalized)',
        # Unvoiced segment features
        'spectralFluxUV_sma3nz_amean': 'Average Spectral Flux (Unvoiced)',
        # Prosodic and temporal features
        'loudnessPeaksPerSec': 'Loudness Peaks Per Second',
        'VoicedSegmentsPerSec': 'Voiced Segments Per Second',
        'MeanVoicedSegmentLengthSec': 'Mean Voiced Segment Length (seconds)',
        'StddevVoicedSegmentLengthSec': 'Standard Deviation of Voiced Segment Length (seconds)',
        'MeanUnvoicedSegmentLength': 'Mean Unvoiced Segment Length (seconds)',
        'StddevUnvoicedSegmentLength': 'Standard Deviation of Unvoiced Segment Length (seconds)',
        'equivalentSoundLevel_dBp': 'Equivalent Sound Level (dBp)',
    }
    
    def __init__(self):
        super().__init__()
        print("Initializing OpenSMILE eGeMAPSv02 extractor...")
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
        # Get feature names by processing a dummy signal
        dummy_signal = np.zeros(16000)  # 1 second at 16kHz
        dummy_features = self.smile.process_signal(dummy_signal, 16000)
        self.feature_names = dummy_features.columns.tolist()
        self.num_features = len(self.feature_names)
        print(f"Loaded {self.num_features} features")
    
    def extract_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract features from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary mapping feature names to values
        """
        features = self.smile.process_file(audio_path)
        # Convert to dictionary (features is a pandas DataFrame with one row)
        return features.iloc[0].to_dict()
    
    def extract_features_from_array(self, audio_array: np.ndarray, 
                                   sample_rate: int = 16000) -> Dict[str, float]:
        """
        Extract features from audio array.
        
        Args:
            audio_array: Audio signal as numpy array (1D, float)
            sample_rate: Sample rate in Hz (default: 16000)
        
        Returns:
            Dictionary mapping feature names to values
        """
        features = self.smile.process_signal(audio_array, sample_rate)
        # Convert to dictionary (features is a pandas DataFrame with one row)
        return features.iloc[0].to_dict()
    
    def features_to_text(self, features: Dict[str, float], precision: int = 3) -> str:
        """
        Convert features to textual description.
        
        Args:
            features: Dictionary of feature name -> value
            precision: Number of decimal places for feature values
        
        Returns:
            Text description of all features
        """
        lines = [
            f"Here are {self.num_features} speech features of the current speaker's sentence.",
            "The features are:"
        ]
        
        for i, (feature_name, value) in enumerate(features.items(), 1):
            # Get human-readable name or use original
            readable_name = self.FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
            formatted_value = f"{value:.{precision}f}"
            
            # Add feature description
            if i < len(features):
                lines.append(f"{readable_name}: {formatted_value},")
            else:
                lines.append(f"{readable_name}: {formatted_value}.")
        
        return " ".join(lines)
    
    def process_file(self, audio_path: str, precision: int = 3) -> str:
        """
        Extract features and generate text description in one step.
        
        Args:
            audio_path: Path to audio file
            precision: Number of decimal places for feature values
        
        Returns:
            Text description of features
        """
        features = self.extract_features(audio_path)
        return self.features_to_text(features, precision)
    
    def process_array(self, audio_array: np.ndarray, sample_rate: int = 16000, 
                     precision: int = 3) -> str:
        """
        Extract features from audio array and generate text description.
        
        Args:
            audio_array: Audio signal as numpy array (1D, float)
            sample_rate: Sample rate in Hz (default: 16000)
            precision: Number of decimal places for feature values
        
        Returns:
            Text description of features
        """
        features = self.extract_features_from_array(audio_array, sample_rate)
        return self.features_to_text(features, precision)
    
    def process_jsonl(self, jsonl_path: str, output_txt_path: str, 
                     max_samples: Optional[int] = None):
        """
        Process all audio files in JSONL and save text descriptions.
        
        Args:
            jsonl_path: Path to JSONL file
            output_txt_path: Path to output text file
            max_samples: Maximum number of samples to process (None = all)
        """
        print(f"Processing JSONL: {jsonl_path}")
        
        with open(jsonl_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        if max_samples:
            data = data[:max_samples]
        
        with open(output_txt_path, 'w') as out_f:
            for i, item in enumerate(data, 1):
                audio_path = item['audio_path']
                label = item.get('label', 'unknown')
                
                print(f"[{i}/{len(data)}] Processing: {audio_path}")
                
                try:
                    feature_text = self.process_file(audio_path)
                    
                    # Write to file
                    out_f.write(f"=== Sample {i} ===\n")
                    out_f.write(f"Audio: {audio_path}\n")
                    out_f.write(f"Label: {label}\n")
                    out_f.write(f"{feature_text}\n\n")
                    
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    continue
        
        print(f"Saved text descriptions to: {output_txt_path}")


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = OpenSMILETextExtractor()
    
    # Process single file
    audio_path = "/netscratch/yelkheir/DeepFense/noise/RIRS_NOISES/pointsource_noises/noise-free-sound-0000.wav"
    text_description = extractor.process_file(audio_path, precision=3)
    print(text_description)
    
    # Process audio array
    import librosa
    audio_array, sr = librosa.load(audio_path, sr=16000)
    text_description = extractor.process_array(audio_array, sample_rate=sr, precision=3)
    print(text_description)
    