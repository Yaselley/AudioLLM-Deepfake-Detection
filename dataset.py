import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
import logging
import librosa

class AudioDeepfakeDataset(Dataset):
    def __init__(self, json_path, sample_rate=16000):
        super().__init__()
        
        # Load data from JSON/JSONL
        self.data = []
        with open(json_path, 'r') as f:
            if json_path.endswith('.json'):
                content = json.load(f)
                if isinstance(content, dict) and "annotation" in content:
                    self.data = content["annotation"]
                else:
                    self.data = content
            else:
                for line in f:
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line in {json_path}")
        
        self.sample_rate = sample_rate
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        audio_path = item["path"]
        label_str = item.get("label", "real") # "real" or "fake"
        
        # Load Audio
        try:
            # Use librosa to ensure resampling is handled robustly
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            logging.error(f"Error reading {audio_path}: {e}")
            audio = np.zeros(self.sample_rate) # 1 sec silence
            
        # Convert to Tensor
        raw_wav = torch.from_numpy(audio).float()
        
        # Text Label
        text_target = "Real" if label_str.lower() == "real" else "Fake"
        
        # Prompt
        prompt = "User: <SpeechHere> Is this audio real or fake? Answer: "
        
        return {
            "raw_wav": raw_wav,
            "text": text_target,
            "prompt": prompt,
            "path": audio_path
        }

    def collate_fn(self, batch):
        # Pad raw audio
        raw_wavs = [b["raw_wav"] for b in batch]
        raw_wav_lens = torch.tensor([len(w) for w in raw_wavs])
        
        # Pad with 0
        padded_wavs = pad_sequence(raw_wavs, batch_first=True, padding_value=0.0)
        
        texts = [b["text"] for b in batch]
        prompts = [b["prompt"] for b in batch]
        paths = [b["path"] for b in batch]
        
        # Create attention mask for audio (1 for real, 0 for pad)
        # Though most models handle this via internal masking or we pass input_values/mask
        audio_attention_mask = torch.arange(padded_wavs.size(1)).unsqueeze(0) < raw_wav_lens.unsqueeze(1)
        
        return {
            "raw_wav": padded_wavs,
            "audio_attention_mask": audio_attention_mask,
            "text": texts,
            "prompt": prompts,
            "path": paths
        }
