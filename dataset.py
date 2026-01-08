import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
import logging
import librosa
import random
import pandas as pd

class AudioDeepfakeDataset(Dataset):
    def __init__(self, json_path, prompt_path=None, sample_rate=16000):
        super().__init__()
        
        # Load data from JSON/JSONL
        self.data = []
        with open(json_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    line = line.strip()
                    self.data.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.error(f"Skipping invalid JSON line in {json_path}")
                
        # Load Prompts
        self.prompts = [
            "User: <SpeechHere> Is this audio real or fake? Answer: "
        ]
        if prompt_path:
            try:
                with open(prompt_path, 'r') as f:
                    prompt_data = json.load(f)
                    if "deepfake_detection" in prompt_data:
                        self.prompts = prompt_data["deepfake_detection"]
            except Exception as e:
                logging.error(f"Error loading prompts from {prompt_path}: {e}")
        
        self.sample_rate = sample_rate
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        audio_path = item["path"]
        label_str = item.get("label", "real") # "real" or "fake"
        
        # Load Audio
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            logging.error(f"Error reading {audio_path}: {e}")
            audio = np.zeros(self.sample_rate) # 1 sec silence
            
        # Convert to Tensor
        raw_wav = torch.from_numpy(audio).float()
        
        # Text Label
        text_target = "Real" if label_str.lower() == "real" else "Fake"
        
        # Select Random Prompt
        prompt = random.choice(self.prompts)
        
        return {
            "raw_wav": raw_wav,
            "text": text_target,
            "prompt": prompt,
            "path": audio_path
        }

    def collate_fn(self, batch):
        raw_wavs = [b["raw_wav"] for b in batch]
        raw_wav_lens = torch.tensor([len(w) for w in raw_wavs])
        
        padded_wavs = pad_sequence(raw_wavs, batch_first=True, padding_value=0.0)
        
        texts = [b["text"] for b in batch]
        prompts = [b["prompt"] for b in batch]
        paths = [b["path"] for b in batch]
        
        audio_attention_mask = torch.arange(padded_wavs.size(1)).unsqueeze(0) < raw_wav_lens.unsqueeze(1)
        
        return {
            "raw_wav": padded_wavs,
            "audio_attention_mask": audio_attention_mask,
            "text": texts,
            "prompt": prompts,
            "path": paths
        }
