"""
Whisper encoder implementation.
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Tuple, Optional

from transformers import WhisperFeatureExtractor
from .base import BaseSpeechEncoder

try:
    from models.modeling_whisper import WhisperModel
except ImportError:
    from transformers import WhisperModel


class WhisperEncoder(BaseSpeechEncoder):
    """Whisper encoder for continuous speech embeddings."""
    
    def __init__(
        self,
        encoder_path: str,
        use_ner: bool = False,
        freeze: bool = True,
        feature_extractor_path: str = "openai/whisper-large-v3"
    ):
        super().__init__(freeze=freeze)
        
        if use_ner:
            from models.whisper_ner import Whisper, ModelDimensions
            
            dims = ModelDimensions(
                n_mels=128, n_audio_ctx=1500, n_audio_state=1280,
                n_audio_head=20, n_audio_layer=32, n_vocab=51865,
                n_text_ctx=448, n_text_state=1280, n_text_head=20, n_text_layer=32
            )
            
            self.encoder = Whisper(dims).encoder
            
            if encoder_path and os.path.exists(encoder_path):
                logging.info(f"Loading WhisperNER weights from {encoder_path}")
                checkpoint = torch.load(encoder_path, map_location="cpu")
                new_state_dict = {}
                for k, v in checkpoint.items():
                    if k.startswith("model.encoder."):
                        new_state_dict[k.replace("model.encoder.", "")] = v
                try:
                    self.encoder.load_state_dict(new_state_dict, strict=True)
                except RuntimeError:
                    self.encoder.load_state_dict(new_state_dict, strict=False)
            
            self._output_dim = 1280
        else:
            self.encoder = WhisperModel.from_pretrained(encoder_path).encoder
            self._output_dim = self.encoder.config.d_model
        
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(feature_extractor_path)
        
        if freeze:
            self.freeze_encoder()
    
    def encode(self, raw_wav: torch.Tensor, audio_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = raw_wav.device
        
        if audio_attention_mask is not None:
            input_lengths = audio_attention_mask.sum(dim=1)
        else:
            input_lengths = torch.tensor([raw_wav.shape[1]] * raw_wav.shape[0], device=device)
        
        with torch.no_grad():
            raw_wav_cpu = raw_wav.cpu().numpy()
            inputs = self.feature_extractor(
                [w for w in raw_wav_cpu], sampling_rate=16000, return_tensors="pt",
                padding="max_length", max_length=480000, truncation=True
            )
            input_features = inputs.input_features.to(device)
        
        outputs = self.encoder(input_features)
        hidden_states = outputs if isinstance(outputs, torch.Tensor) else outputs.last_hidden_state
        
        lengths = self.get_feat_extract_output_lengths(input_lengths)
        lengths = torch.clamp(lengths, max=1500)
        
        return hidden_states, lengths
    
    def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        feat_lens = (input_lengths / 160).ceil().long()
        if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
            return self.encoder._get_feat_extract_output_lengths(feat_lens)
        return (feat_lens + 1) // 2
