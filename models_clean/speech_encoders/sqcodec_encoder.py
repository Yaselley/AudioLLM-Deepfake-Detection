"""
SQ-Codec encoder implementation.
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .base import BaseSpeechEncoder


class SQCodecEncoder(BaseSpeechEncoder):
    """
    SQ-Codec encoder for discrete tokens.
    """
    
    def __init__(
        self,
        encoder_path: str,
        num_quantizers: Optional[int] = None,
        num_levels: Optional[int] = None,
        freeze: bool = True
    ):
        """
        Args:
            encoder_path: Path to SQ-Codec model directory or model name
            num_quantizers: Number of quantizers (auto-detected if None)
            num_levels: Number of quantization levels (auto-detected if None)
            freeze: Whether to freeze encoder parameters (always True for SQ-Codec)
        """
        super().__init__(freeze=True)  # SQ-Codec is always frozen
        
        # Load SQ-Codec
        try:
            import sq_codec
        except ImportError:
            try:
                import sqcodec as sq_codec
            except ImportError as e:
                if "Self" in str(e):
                    raise ImportError(
                        "SQ-Codec requires Python 3.11+ or typing_extensions. "
                        "Install typing_extensions: pip install typing_extensions"
                    )
                raise ImportError("SQ-Codec not found. Please install: pip install sq-codec")
        
        # Determine model path
        if os.path.isdir(encoder_path):
            model_path = encoder_path
        else:
            model_path = encoder_path
        
        logging.info(f"Loading SQ-Codec from {model_path}")
        
        # Load model
        if hasattr(sq_codec, "get_model"):
            try:
                self.encoder = sq_codec.get_model(model_path)
            except:
                if os.path.exists(model_path) or os.path.isdir(model_path):
                    if hasattr(sq_codec, "SQCodec") and hasattr(sq_codec.SQCodec, "from_pretrained"):
                        self.encoder = sq_codec.SQCodec.from_pretrained(model_path)
                    else:
                        raise ValueError(f"Could not load SQ-Codec from {model_path}")
                else:
                    raise ValueError(f"SQ-Codec model '{model_path}' not found")
        elif hasattr(sq_codec, "SQCodec"):
            if os.path.exists(model_path) or os.path.isdir(model_path):
                self.encoder = sq_codec.SQCodec.from_pretrained(model_path)
            else:
                raise ValueError(f"SQ-Codec path not found: {model_path}")
        else:
            raise ImportError("SQ-Codec module structure not recognized")
        
        # Freeze encoder
        if hasattr(self.encoder, "network"):
            self.encoder.network.eval()
            for param in self.encoder.network.parameters():
                param.requires_grad = False
        else:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get codebook info
        codec_model = self.encoder
        if hasattr(self.encoder, "network"):
            codec_model = self.encoder.network
        
        if num_quantizers is None:
            if hasattr(codec_model, "num_quantizers"):
                self.num_quantizers = codec_model.num_quantizers
            elif hasattr(codec_model, "n_q"):
                self.num_quantizers = codec_model.n_q
            elif hasattr(self.encoder, "config"):
                if hasattr(self.encoder.config, "num_quantizers"):
                    self.num_quantizers = self.encoder.config.num_quantizers
                elif hasattr(self.encoder.config, "n_q"):
                    self.num_quantizers = self.encoder.config.n_q
                else:
                    self.num_quantizers = 4  # Default
            else:
                self.num_quantizers = 4  # Default
        else:
            self.num_quantizers = num_quantizers
        
        if num_levels is None:
            if hasattr(codec_model, "num_levels"):
                self.num_levels = codec_model.num_levels
            elif hasattr(codec_model, "codebook_size"):
                self.num_levels = codec_model.codebook_size
            elif hasattr(self.encoder, "config"):
                if hasattr(self.encoder.config, "num_levels"):
                    self.num_levels = self.encoder.config.num_levels
                elif hasattr(self.encoder.config, "codebook_size"):
                    self.num_levels = self.encoder.config.codebook_size
                else:
                    self.num_levels = 4096  # Default
            else:
                self.num_levels = 4096  # Default
        else:
            self.num_levels = num_levels
        
        logging.info(f"SQ-Codec: num_quantizers={self.num_quantizers}, num_levels={self.num_levels}")
        
        # Discrete tokens: output_dim is None
        self._output_dim = None
    
    def encode(self, raw_wav: torch.Tensor, audio_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio using SQ-Codec.
        
        Args:
            raw_wav: [B, T] raw audio
            audio_attention_mask: [B, T] attention mask
        
        Returns:
            codes: [B, n_q, T] discrete code values
            lengths: [B] valid lengths
        """
        device = raw_wav.device
        
        # Calculate input lengths
        if audio_attention_mask is not None:
            input_lengths = audio_attention_mask.sum(dim=1)
        else:
            input_lengths = torch.tensor([raw_wav.shape[1]] * raw_wav.shape[0], device=device)
        
        # Encode to codes
        with torch.no_grad():
            if raw_wav.dtype != torch.float32:
                raw_wav_input = raw_wav.to(torch.float32)
            else:
                raw_wav_input = raw_wav
            
            # SQ-Codec encoding
            if hasattr(self.encoder, "encode_audio"):
                result = self.encoder.encode_audio(raw_wav_input)
                if isinstance(result, tuple):
                    codes = result[1]  # indices: [n_q, B, T] or [B, n_q, T]
                else:
                    codes = result
            elif hasattr(self.encoder, "encode"):
                codes = self.encoder.encode(raw_wav_input)
            else:
                codes = self.encoder(raw_wav_input)
                if isinstance(codes, tuple):
                    codes = codes[1]
            
            # Ensure codes are [B, n_q, T] format
            if codes.dim() == 3:
                if codes.shape[0] == self.num_quantizers:
                    codes = codes.transpose(0, 1)  # [n_q, B, T] -> [B, n_q, T]
                elif codes.shape[1] == self.num_quantizers:
                    pass  # Already [B, n_q, T]
                else:
                    if codes.shape[2] == self.num_quantizers:
                        codes = codes.transpose(1, 2)  # [B, T, n_q] -> [B, n_q, T]
                    else:
                        raise ValueError(f"Unexpected SQ-Codec output shape: {codes.shape}")
            elif codes.dim() == 2:
                if self.num_quantizers == 1:
                    codes = codes.unsqueeze(1)  # [B, T] -> [B, 1, T]
                else:
                    raise ValueError(f"SQ-Codec has {self.num_quantizers} quantizers but got 2D codes")
            else:
                raise ValueError(f"Unexpected SQ-Codec output shape: {codes.shape}")
            
            # Slice to requested number of quantizers
            if codes.shape[1] > self.num_quantizers:
                codes = codes[:, :self.num_quantizers, :]
        
        # Calculate output lengths
        batch_size = codes.shape[0]
        max_code_len = codes.shape[2]
        
        if audio_attention_mask is not None:
            # SQ-Codec stride: typically 320 samples
            speech_lens = (input_lengths / 320).ceil().long()
            speech_lens = torch.clamp(speech_lens, max=max_code_len)
        else:
            input_lengths = torch.tensor([raw_wav.shape[1]] * batch_size, device=device)
            speech_lens = (input_lengths / 320).ceil().long()
            speech_lens = torch.clamp(speech_lens, max=max_code_len)
        
        return codes, speech_lens
    
    def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute output lengths for SQ-Codec.
        SQ-Codec stride is typically 320 samples.
        """
        return (input_lengths / 320).ceil().long()
