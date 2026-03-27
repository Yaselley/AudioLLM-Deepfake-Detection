"""
Dataset for Thinking-mode deepfake detection training.

Input JSONL format:
  {"path": "/path/to/audio.flac", "label": "real", "thinking": "<think>...</think>\n<answer>Real</answer>"}

Training target (samples["text"]):
  - If "thinking" field exists → full thinking string (model learns to produce the chain-of-thought)
  - If absent → simple "Real"/"Fake" derived from "label" (fallback for val/test)

The "label" field is metadata only; it is NOT used as the training target when "thinking" is present.
"""

import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import logging
import librosa
import random

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")
warnings.filterwarnings("ignore", category=UserWarning)


class AudioDeepfakeThinkingDataset(Dataset):
    def __init__(self, json_path, prompt_path=None, system_prompt_path=None, sample_rate=16000):
        super().__init__()

        self.data = []
        with open(json_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logging.error(f"Skipping invalid JSON line in {json_path}")

        self.prompts = [
            "User: <SpeechHere> Is this audio real or fake? Answer: "
        ]
        if prompt_path:
            try:
                with open(prompt_path, 'r') as f:
                    prompt_data = json.load(f)
                    if "deepfake_detection" in prompt_data:
                        self.prompts = prompt_data["deepfake_detection"]
                logging.info(f"Loaded prompts from {prompt_path}: {self.prompts}")
            except Exception as e:
                logging.error(f"Error loading prompts from {prompt_path}: {e}")

        self.system_prompt = None
        if system_prompt_path:
            try:
                with open(system_prompt_path, 'r') as f:
                    self.system_prompt = f.read().strip()
                logging.info(f"Loaded system prompt from {system_prompt_path} ({len(self.system_prompt)} chars)")
            except Exception as e:
                logging.error(f"Error loading system prompt: {e}")

        self.sample_rate = sample_rate
        self.max_audio_sec = 10
        self.max_audio_len = int(self.sample_rate * self.max_audio_sec)

        # Stats
        n_thinking = sum(1 for d in self.data if "thinking" in d and d["thinking"])
        logging.info(f"Loaded {len(self.data)} samples from {json_path} ({n_thinking} with thinking, {len(self.data) - n_thinking} without)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        audio_path = item["path"]

        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            logging.error(f"Error reading {audio_path}: {e}")
            audio = np.zeros(self.sample_rate)

        if audio.shape[0] > self.max_audio_len:
            audio = audio[:self.max_audio_len]

        raw_wav = torch.from_numpy(audio).float()

        # Target: full thinking string if available, else simple label
        if "thinking" in item and item["thinking"]:
            text_target = item["thinking"]
        else:
            label_str = item.get("label", "real")
            text_target = "Real" if label_str.lower() == "real" else "Fake"

        prompt = random.choice(self.prompts)

        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{prompt}"

        return {
            "raw_wav": raw_wav,
            "text": text_target,
            "prompt": prompt,
            "path": audio_path,
        }

    def collate_fn(self, batch):
        raw_wavs = [b["raw_wav"] for b in batch]
        raw_wav_lens = torch.tensor([len(w) for w in raw_wavs])

        padded_wavs = pad_sequence(raw_wavs, batch_first=True, padding_value=0.0)

        texts = [b["text"] for b in batch]
        prompts = [b["prompt"] for b in batch]
        paths = [b["path"] for b in batch]

        attention_mask = torch.arange(padded_wavs.size(1)).unsqueeze(0) < raw_wav_lens.unsqueeze(1)
        padding_mask = ~attention_mask

        return {
            "raw_wav": padded_wavs,
            "padding_mask": padding_mask,
            "audio_attention_mask": attention_mask,
            "text": texts,
            "prompt": prompts,
            "path": paths,
        }
