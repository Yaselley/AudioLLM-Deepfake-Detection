"""
Wav2Vec2 encoder implementation.
"""

import logging
import torch
from typing import Tuple, Optional

from transformers import Wav2Vec2Model, Wav2Vec2BertModel, AutoFeatureExtractor
from .base import BaseSpeechEncoder


class Wav2Vec2Encoder(BaseSpeechEncoder):
    """Wav2Vec2 encoder for continuous speech embeddings."""
    
    def __init__(
        self,
        encoder_path: str,
        encoder_source: str = "huggingface",
        use_bert: bool = False,
        freeze: bool = True
    ):
        super().__init__(freeze=freeze)
        
        self.encoder_source = encoder_source.lower()
        self.use_bert = use_bert
        
        if use_bert:
            self.encoder = Wav2Vec2BertModel.from_pretrained(encoder_path)
            self._output_dim = self.encoder.config.hidden_size
            self.processor = AutoFeatureExtractor.from_pretrained(encoder_path)
        elif self.encoder_source == "fairseq":
            import fairseq
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([encoder_path])
            self.encoder = models[0]
            self._output_dim = cfg.model.encoder_embed_dim
        else:
            self.encoder = Wav2Vec2Model.from_pretrained(encoder_path)
            self._output_dim = self.encoder.config.hidden_size
        
        if freeze:
            self.freeze_encoder()
    
    def encode(self, raw_wav: torch.Tensor, audio_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = raw_wav.device
        
        if audio_attention_mask is not None:
            input_lengths = audio_attention_mask.sum(dim=1)
        else:
            input_lengths = torch.tensor([raw_wav.shape[1]] * raw_wav.shape[0], device=device)
        
        if self.encoder_source == "fairseq":
            if raw_wav.dtype != torch.float32:
                raw_wav_input = raw_wav.to(torch.float32)
            else:
                raw_wav_input = raw_wav
            res = self.encoder(raw_wav_input, mask=False, features_only=True)
            hidden_states = res["x"]
        elif self.use_bert:
            with torch.no_grad():
                raw_wav_cpu = raw_wav.cpu().numpy()
                inputs = self.processor([w for w in raw_wav_cpu], sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
            outputs = self.encoder(input_values)
            hidden_states = outputs.last_hidden_state
        else:
            outputs = self.encoder(raw_wav)
            hidden_states = outputs.last_hidden_state
        
        lengths = self.get_feat_extract_output_lengths(input_lengths)
        return hidden_states, lengths
    
    def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        if self.use_bert:
            feat_lens = (input_lengths / 160).ceil().long()
            if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
                return self.encoder._get_feat_extract_output_lengths(feat_lens)
            return (feat_lens + 3) // 4
        elif hasattr(self.encoder, "_get_feat_extract_output_lengths"):
            return self.encoder._get_feat_extract_output_lengths(input_lengths)
        else:
            return (input_lengths / 320).ceil().long()
