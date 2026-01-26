"""
Base class for all speech encoders.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseSpeechEncoder(nn.Module, ABC):
    """
    Base class for all speech encoders.
    
    All encoders should:
    1. Encode raw audio to either continuous embeddings or discrete tokens
    2. Support freezing/unfreezing parameters
    3. Provide output dimension information
    4. Handle variable-length sequences
    """
    
    def __init__(self, freeze: bool = True):
        """
        Args:
            freeze: Whether to freeze encoder parameters (default: True)
        """
        super().__init__()
        self.freeze = freeze
        self._output_dim = None  # Will be set by subclasses
    
    @property
    def output_dim(self) -> Optional[int]:
        """
        Returns the output dimension of the encoder.
        Returns None for discrete token encoders (e.g., SpeechTokenizer, SQ-Codec).
        """
        return self._output_dim
    
    @property
    def is_discrete(self) -> bool:
        """
        Returns True if encoder outputs discrete tokens, False if continuous embeddings.
        """
        return self._output_dim is None
    
    @abstractmethod
    def encode(self, raw_wav: torch.Tensor, audio_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode raw audio to embeddings or discrete tokens.
        
        Args:
            raw_wav: Raw audio waveform [B, T] (float32, normalized to [-1, 1])
            audio_attention_mask: Optional attention mask [B, T] (1 for valid, 0 for padding)
        
        Returns:
            output: Encoded representation
                - For continuous encoders: [B, T_out, D] embeddings
                - For discrete encoders: [B, n_layers, T_out] discrete codes
            lengths: Valid lengths [B] (int tensor)
        """
        pass
    
    @abstractmethod
    def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute output sequence lengths from input audio lengths.
        
        Args:
            input_lengths: Input audio lengths [B] (int tensor)
        
        Returns:
            output_lengths: Output sequence lengths [B] (int tensor)
        """
        pass
    
    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        self.freeze = True
    
    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        self.freeze = False
