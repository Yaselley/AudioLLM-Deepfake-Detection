"""
SpeechTokenizer encoder implementation.
Supports both discrete tokens and continuous embeddings.
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .base import BaseSpeechEncoder


class SpeechTokenizerEncoder(BaseSpeechEncoder):
    """
    SpeechTokenizer encoder.
    
    Can return either:
    1. Discrete tokens: [B, n_q, T] discrete code values
    2. Continuous embeddings: [B, T, D] continuous embeddings from quantizer
    """
    
    def __init__(
        self,
        encoder_path: str,
        return_discrete_tokens: bool = True,
        num_layers: Optional[int] = None,
        codebook_size: Optional[int] = None,
        freeze: bool = True
    ):
        """
        Args:
            encoder_path: Path to SpeechTokenizer model directory or config file
            return_discrete_tokens: If True, return discrete codes [B, n_q, T]
                                   If False, return continuous embeddings [B, T, D]
            num_layers: Number of RVQ layers (auto-detected if None)
            codebook_size: Codebook size (auto-detected if None)
            freeze: Whether to freeze encoder parameters (always True for SpeechTokenizer)
        """
        super().__init__(freeze=True)  # SpeechTokenizer is always frozen
        
        # Load SpeechTokenizer
        try:
            from speechtokenizer import SpeechTokenizer
        except ImportError:
            try:
                from speechtokenizer.speechtokenizer import SpeechTokenizer
            except ImportError:
                raise ImportError("SpeechTokenizer not found. Please install: pip install speechtokenizer")
        
        # Determine config_path and ckpt_path
        if os.path.isdir(encoder_path):
            config_path = os.path.join(encoder_path, "config.json")
            pt_files = [f for f in os.listdir(encoder_path) if f.endswith('.pt')]
            if pt_files:
                ckpt_path = os.path.join(encoder_path, pt_files[0])
            else:
                raise FileNotFoundError(f"No .pt file found in {encoder_path}")
        else:
            config_path = encoder_path
            audio_encoder_dir = os.path.dirname(encoder_path)
            pt_files = [f for f in os.listdir(audio_encoder_dir) if f.endswith('.pt')]
            if pt_files:
                ckpt_path = os.path.join(audio_encoder_dir, pt_files[0])
            else:
                raise FileNotFoundError(f"No .pt file found in {audio_encoder_dir}")
        
        logging.info(f"Loading SpeechTokenizer from config: {config_path}, checkpoint: {ckpt_path}")
        self.encoder = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Get codebook info from the loaded model
        # SpeechTokenizer stores n_q and codebook_size at the model level
        if hasattr(self.encoder, "n_q"):
            detected_num_layers = self.encoder.n_q
        elif hasattr(self.encoder, "config") and hasattr(self.encoder.config, "n_q"):
            detected_num_layers = self.encoder.config.n_q
        else:
            detected_num_layers = None
        
        if hasattr(self.encoder, "quantizer") and hasattr(self.encoder.quantizer, "bins"):
            detected_codebook_size = self.encoder.quantizer.bins
        elif hasattr(self.encoder, "config") and hasattr(self.encoder.config, "codebook_size"):
            detected_codebook_size = self.encoder.config.codebook_size
        else:
            detected_codebook_size = None
        
        # Use detected values or fall back to user-provided values
        if num_layers is not None:
            self.num_layers = num_layers
            logging.info(f"Using user-specified num_layers: {self.num_layers}")
        elif detected_num_layers is not None:
            self.num_layers = detected_num_layers
            logging.info(f"Auto-detected num_layers from model: {self.num_layers}")
        else:
            raise ValueError("Could not determine num_layers. Please specify manually.")
        
        if codebook_size is not None:
            self.codebook_size = codebook_size
            logging.info(f"Using user-specified codebook_size: {self.codebook_size}")
        elif detected_codebook_size is not None:
            self.codebook_size = detected_codebook_size
            logging.info(f"Auto-detected codebook_size from model: {self.codebook_size}")
        else:
            raise ValueError("Could not determine codebook_size. Please specify manually.")
        
        logging.info(f"SpeechTokenizer: {self.num_layers} layers, codebook size {self.codebook_size}")
        
        # Set output mode
        self.return_discrete_tokens = return_discrete_tokens
        
        # CRITICAL FIX: Set is_discrete property
        self._is_discrete = return_discrete_tokens
        
        if return_discrete_tokens:
            # Discrete mode: output_dim is None
            self._output_dim = None
        else:
            # Continuous mode: get embedding dimension from quantizer
            # SpeechTokenizer quantizer typically has embedding_dim
            if hasattr(self.encoder, "quantizer") and hasattr(self.encoder.quantizer, "embed_dim"):
                self._output_dim = self.encoder.quantizer.embed_dim
            elif hasattr(self.encoder, "quantizer") and hasattr(self.encoder.quantizer, "dim"):
                self._output_dim = self.encoder.quantizer.dim
            else:
                # Default fallback (typically 1024 for SpeechTokenizer)
                self._output_dim = 1024
                logging.warning(f"Could not determine embedding dimension, using default: {self._output_dim}")
    
    @property
    def is_discrete(self) -> bool:
        """Return whether this encoder outputs discrete tokens."""
        return self._is_discrete
    
    @property
    def output_dim(self) -> Optional[int]:
        """Return output dimension (None for discrete encoders)."""
        return self._output_dim
    
    def encode(self, raw_wav: torch.Tensor, audio_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio using SpeechTokenizer.
        
        Args:
            raw_wav: [B, T] raw audio
            audio_attention_mask: [B, T] attention mask
        
        Returns:
            If return_discrete_tokens=True:
                codes: [B, n_q, T] discrete code values
            If return_discrete_tokens=False:
                embeddings: [B, T, D] continuous embeddings (post-quantization)
            lengths: [B] valid lengths
        """
        device = raw_wav.device
        
        # Calculate input lengths
        if audio_attention_mask is not None:
            input_lengths = audio_attention_mask.sum(dim=1)
        else:
            input_lengths = torch.tensor([raw_wav.shape[1]] * raw_wav.shape[0], device=device)
        
        # Prepare input
        with torch.no_grad():
            if raw_wav.dtype != torch.float32:
                raw_wav_input = raw_wav.to(torch.float32)
            else:
                raw_wav_input = raw_wav
            
            # SpeechTokenizer expects [B, 1, T]
            wav_input = raw_wav_input.unsqueeze(1)
            
            if self.return_discrete_tokens:
                # Discrete mode: use encode() to get codes
                # encode() returns [n_q, B, T]
                codes = self.encoder.encode(wav_input, n_q=self.num_layers)  # [n_q, B, T]
                codes = codes.transpose(0, 1)  # [B, n_q, T]
                
                # Calculate output lengths
                batch_size = codes.shape[0]
                max_code_len = codes.shape[2]
                
                # SpeechTokenizer stride: typically 320 samples per code
                speech_lens = (input_lengths / 320).ceil().long()
                speech_lens = torch.clamp(speech_lens, max=max_code_len)
                
                return codes, speech_lens
            else:
                # Continuous mode: use forward_feature() to get post-quantization embeddings
                # This respects num_layers and returns quantized embeddings from specified layers
                layers = list(range(self.num_layers))
                quantized_list = self.encoder.forward_feature(wav_input, layers=layers)
                # quantized_list is a list of [B, D, T] tensors, one per layer
                # We need to aggregate them
                if len(quantized_list) == 1:
                    # Single layer: just use it
                    embeddings = quantized_list[0]  # [B, D, T]
                else:
                    # Multiple layers: sum them (residual structure)
                    # Stack and sum across layers
                    embeddings = torch.stack(quantized_list, dim=0).sum(dim=0)  # [B, D, T]

                embeddings = embeddings.transpose(1, 2)  # [B, T, D]
                
                # Calculate output lengths
                batch_size = embeddings.shape[0]
                max_emb_len = embeddings.shape[1]
                
                # SpeechTokenizer stride: typically 320 samples per frame
                speech_lens = (input_lengths / 320).ceil().long()
                speech_lens = torch.clamp(speech_lens, max=max_emb_len)
                
                return embeddings, speech_lens
    
    def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute output lengths for SpeechTokenizer.
        SpeechTokenizer stride is typically 320 samples.
        """
        return (input_lengths / 320).ceil().long()