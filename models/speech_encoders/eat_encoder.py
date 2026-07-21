"""
EAT (Efficient Audio Transformer) encoder implementation.
"""

import torch
from typing import Tuple, Optional

from .base import BaseSpeechEncoder
import sys 
# Add DeepFense path for custom modules if needed
sys.path.append("/netscratch/yelkheir/DeepFense/DeepFense")


class EATEncoder(BaseSpeechEncoder):
    """EAT encoder for continuous speech embeddings."""

    def __init__(
        self,
        encoder_path: str,
        encoder_source: str = "huggingface",
        freeze: bool = True,
        trust_remote_code: bool = True,
    ):
        super().__init__(freeze=freeze)

        # EAT is provided by DeepFense as a frontend wrapper.
        from deepfense.models.frontends.eat import EATWrapper

        eat_config = {
            "source": encoder_source,
            "ckpt_path": encoder_path,
            "freeze": freeze,
            "trust_remote_code": trust_remote_code,
        }
        self.encoder = EATWrapper(eat_config)

        # Best-effort inference of output dim (EAT-large is typically 1024).
        dim = None
        cfg = getattr(getattr(self.encoder, "model", None), "config", None)
        for attr in ("embed_dim", "hidden_size", "d_model"):
            if cfg is not None and hasattr(cfg, attr):
                dim = getattr(cfg, attr)
                break
        self._output_dim = int(dim) if dim is not None else 1024

        if freeze:
            # EATWrapper freezes its internal HF model; we also set eval() for consistency.
            self.freeze_encoder()

    def encode(
        self,
        raw_wav: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = raw_wav.device

        if audio_attention_mask is not None:
            input_lengths = audio_attention_mask.sum(dim=1)
        else:
            input_lengths = torch.tensor([raw_wav.shape[1]] * raw_wav.shape[0], device=device)

        raw_wav_input = raw_wav.to(torch.float32) if raw_wav.dtype != torch.float32 else raw_wav

        hidden_states = self.encoder(raw_wav_input)
        lengths = self.get_feat_extract_output_lengths(input_lengths)

        return hidden_states, lengths

    def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        EAT uses 10ms fbank frames (160 samples at 16kHz), then patching:
        - time patches: 16 frames per patch
        - 8 freq patches + 1 CLS token
        """
        feat_lens = (input_lengths / 160).floor().long()
        t_patches = (feat_lens / 16).floor().long()
        return t_patches * 8 + 1

