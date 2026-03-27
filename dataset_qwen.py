import warnings
# Ignore the specific dependency warning
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import json
import librosa
import logging
import random
import numpy as np
from torch.utils.data import Dataset

class AudioDeepfakeDataset(Dataset):
    def __init__(self, json_path, prompt_path=None, sample_rate=16000):
        super().__init__()
        self.data = []
        with open(json_path, 'r') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
                
        # Load custom prompts or use default
        self.prompts = ["<SpeechHere> Is this audio real or fake?"]
        if prompt_path:
            try:
                with open(prompt_path, 'r') as f:
                    prompt_data = json.load(f)
                    if "deepfake_detection" in prompt_data:
                        self.prompts = prompt_data["deepfake_detection"]
            except Exception as e:
                logging.error(f"Error loading prompts: {e}")
        
        self.sample_rate = sample_rate
        self.max_audio_len = sample_rate * 10 # 10 seconds limit

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        audio_path = item["path"]
        label_str = item.get("label", "real")
        
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            if len(audio) > self.max_audio_len:
                audio = audio[:self.max_audio_len]
        except Exception:
            audio = np.zeros(self.sample_rate)

        # Target response for the LLM
        text_target = "Real" if label_str.lower() == "real" else "Fake"
        prompt = random.choice(self.prompts)
        
        return {
            "audio": audio, # Keep as numpy for the processor
            "text": text_target,
            "prompt": prompt
        }

    def collate_fn(self, batch):
        return {
            "raw_wav": [b["audio"] for b in batch], # List of arrays
            "text": [b["text"] for b in batch],
            "prompt": [b["prompt"] for b in batch]
        }