"""
WavLM encoder implementation.
"""

import logging
import torch
from typing import Tuple, Optional

from transformers import WavLMModel
from .base import BaseSpeechEncoder


class WavLMEncoder(BaseSpeechEncoder):
    """WavLM encoder for continuous speech embeddings."""
    
    def __init__(
        self,
        encoder_path: str,
        encoder_source: str = "huggingface",
        freeze: bool = True
    ):
        super().__init__(freeze=freeze)
        
        self.encoder_source = encoder_source.lower()
        
        if self.encoder_source in ["unil", "microsoft"]:
            from deepfense.models.modules.wavlm.wavlm import WavLM, WavLMConfig
            checkpoint = torch.load(encoder_path)
            cfg = WavLMConfig(checkpoint["cfg"])
            self.encoder = WavLM(cfg)
            self.encoder.load_state_dict(checkpoint["model"], strict=False)
            self._output_dim = cfg.encoder_embed_dim
        else:
            self.encoder = WavLMModel.from_pretrained(encoder_path)
            self._output_dim = self.encoder.config.hidden_size
        
        if freeze:
            self.freeze_encoder()
    
    def encode(self, raw_wav: torch.Tensor, audio_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = raw_wav.device
        
        if audio_attention_mask is not None:
            input_lengths = audio_attention_mask.sum(dim=1)
        else:
            input_lengths = torch.tensor([raw_wav.shape[1]] * raw_wav.shape[0], device=device)
        
        if self.encoder_source in ["unil", "microsoft"]:
            if raw_wav.dtype != torch.float32:
                raw_wav_input = raw_wav.to(torch.float32)
            else:
                raw_wav_input = raw_wav
            res = self.encoder(raw_wav_input, mask=False, features_only=True)
            hidden_states = res["x"]
        else:
            outputs = self.encoder(raw_wav)
            hidden_states = outputs.last_hidden_state
        
        lengths = self.get_feat_extract_output_lengths(input_lengths)
        return hidden_states, lengths
    
    def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
            return self.encoder._get_feat_extract_output_lengths(input_lengths)
        else:
            return (input_lengths / 320).ceil().long()
