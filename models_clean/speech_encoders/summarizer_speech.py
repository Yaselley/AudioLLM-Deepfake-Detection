import opensmile
import numpy as np
from typing import Dict

class SpeechFeatureSummarizer:
    def __init__(self):
        # We still use eGeMAPSv02 as it contains all necessary raw data
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
    def _categorize(self, value: float, low_thresh: float, high_thresh: float) -> str:
        if value < low_thresh: return "low"
        if value > high_thresh: return "high"
        return "medium"

    def extract_and_summarize(self, audio_path: str) -> str:
        # 1. Get raw features
        raw_df = self.smile.process_file(audio_path)
        f = raw_df.iloc[0].to_dict()

        # 2. Extract specific values for paralinguistic features
        # Note: OpenSMILE F0 is often in semitones (F0semitoneFrom27.5Hz)
        # We convert back to Hz for easier gender/pitch reading: Hz = 27.5 * 2^(semitones/12)
        f0_mean_hz = 27.5 * (2 ** (f['F0semitoneFrom27.5Hz_sma3nz_amean'] / 12))
        f0_std = f['F0semitoneFrom27.5Hz_sma3nz_stddevNorm']
        
        loudness = f['loudness_sma3_amean']
        jitter = f['jitterLocal_sma3nz_amean']
        shimmer = f['shimmerLocaldB_sma3nz_amean']
        
        # Speaking rate: based on voiced segments per second
        speaking_rate_raw = f['VoicedSegmentsPerSec']

        # 3. Logic for Gender-Informed F0
        gender = "female" if f0_mean_hz > 165 else "male"
        gender_text = f"The sentence is spoken by a {gender} speaker with F0 value of {f0_mean_hz:.1f} Hz."

        # 4. Logic for Paralinguistic Categories (Thresholds are heuristic)
        summary = {
            "Loudness": self._categorize(loudness, 0.5, 1.5),
            "Pitch": self._categorize(f0_mean_hz, 120, 200),
            "Pitch range": "wide" if f0_std > 0.2 else ("narrow" if f0_std < 0.1 else "medium"),
            "Speaking rate": "fast" if speaking_rate_raw > 3.0 else ("slow" if speaking_rate_raw < 1.5 else "medium"),
            "Jitter": self._categorize(jitter, 0.01, 0.05),
            "Shimmer": self._categorize(shimmer, 0.5, 1.5)
        }

        # 5. Format Output to match your UI
        output = [
            "--- Gender-Informed F0 ---",
            gender_text,
            "",
            "--- Paralinguistic Features ---"
        ]
        for key, val in summary.items():
            output.append(f"{key}: {val}")

        return "\n".join(output)

# Example Usage
summarizer = SpeechFeatureSummarizer()
print(summarizer.extract_and_summarize("/netscratch/yelkheir/fft_test.wav"))