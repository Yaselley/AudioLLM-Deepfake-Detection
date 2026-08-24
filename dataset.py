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

import warnings
# Ignore the specific dependency warning
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")
warnings.filterwarnings("ignore", category=UserWarning)

class AudioDeepfakeDataset(Dataset):
    def __init__(self, json_path, prompt_path=None, sample_rate=16000):
        super().__init__()
        
        # Load data from JSON/JSONL or a list of JSON/JSONL paths
        self.data = []
        paths = json_path if isinstance(json_path, list) else [json_path]
        for path in paths:
            with open(path, 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    self.data.extend(loaded)
                else:
                    self.data.append(loaded)
            
        # Load Prompts
        self.prompts = [
            "User: <SpeechHere> Is this audio real or fake? Answer: "
        ]
        if prompt_path:
            try:
                with open(prompt_path, 'r') as f:
                    prompt_data = json.load(f)
                    if "deepfake_detection_with_reasoning" in prompt_data:
                        self.prompts = prompt_data["deepfake_detection_with_reasoning"]
            except Exception as e:
                logging.error(f"Error loading prompts from {prompt_path}: {e}")
        
        self.sample_rate = sample_rate
        self.max_audio_sec = 10
        self.max_audio_len = int(self.sample_rate * self.max_audio_sec)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        audio_path = item["path"]
        label_str = item.get("text", "Real") # "real" or "fake"
        # Load Audio
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            logging.error(f"Error reading {audio_path}: {e}")
            audio = np.zeros(self.sample_rate) # 1 sec silence

        # Hard cap: keep at most 10 seconds of audio
        if audio.shape[0] > self.max_audio_len:
            audio = audio[: self.max_audio_len]

        # Convert to Tensor
        raw_wav = torch.from_numpy(audio).float()

        # Text Label
        text_target = label_str

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

        # 1 for valid, 0 for padding (Standard Transformers)
        attention_mask = torch.arange(padded_wavs.size(1)).unsqueeze(0) < raw_wav_lens.unsqueeze(1)
        
        # True for padding, False for valid (Common for Fairseq/WavLM)
        padding_mask = ~attention_mask
        
        return {
            "raw_wav": padded_wavs,
            "padding_mask": padding_mask,
            "audio_attention_mask": attention_mask, # 1=valid, 0=padding
            "text": texts,
            "prompt": prompts,
            "path": paths
        }
