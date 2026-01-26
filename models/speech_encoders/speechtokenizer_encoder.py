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
        
        # Get codebook info
        if hasattr(self.encoder, "n_q") and hasattr(self.encoder, "codebook_size"):
            self.num_layers = self.encoder.n_q
            self.codebook_size = self.encoder.codebook_size
        elif hasattr(self.encoder, "config"):
            self.num_layers = getattr(self.encoder.config, "n_q", num_layers)
            self.codebook_size = getattr(self.encoder.config, "codebook_size", codebook_size)
        else:
            self.num_layers = num_layers
            self.codebook_size = codebook_size
        
        if self.num_layers is None or self.codebook_size is None:
            raise ValueError("Could not determine num_layers or codebook_size. Please specify manually.")
        
        logging.info(f"SpeechTokenizer: {self.num_layers} layers, codebook size {self.codebook_size}")
        
        # Set output mode
        self.return_discrete_tokens = return_discrete_tokens
        
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
                embeddings: [B, T, D] continuous embeddings
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
            
            # SpeechTokenizer encode returns [n_q, B, T]
            codes = self.encoder.encode(raw_wav_input.unsqueeze(1))  # [n_q, B, T]
            codes = codes[:self.num_layers, :, :]
            codes = codes.transpose(0, 1)  # [B, n_q, T]
        
        # Calculate output lengths
        batch_size = codes.shape[0]
        max_code_len = codes.shape[2]
        
        if audio_attention_mask is not None:
            # SpeechTokenizer stride: typically 320 samples per code
            speech_lens = (input_lengths / 320).ceil().long()
            speech_lens = torch.clamp(speech_lens, max=max_code_len)
        else:
            input_lengths = torch.tensor([raw_wav.shape[1]] * batch_size, device=device)
            speech_lens = (input_lengths / 320).ceil().long()
            speech_lens = torch.clamp(speech_lens, max=max_code_len)
        
        if self.return_discrete_tokens:
            # Return discrete codes
            return codes, speech_lens
        else:
            # Convert codes to continuous embeddings
            # Get quantizer embeddings
            if hasattr(self.encoder, "quantizer"):
                quantizer = self.encoder.quantizer
            else:
                raise ValueError("SpeechTokenizer encoder does not have quantizer for continuous embeddings")
            
            # codes: [B, n_q, T]
            # We need to get embeddings for each layer
            # For simplicity, we'll use the first layer's embeddings
            # Or average across layers if multiple quantizers
            
            batch_size, n_q, seq_len = codes.shape
            embeddings_list = []
            
            for layer_idx in range(n_q):
                layer_codes = codes[:, layer_idx, :]  # [B, T]
                # Get embeddings from quantizer for this layer
                if hasattr(quantizer, "codebooks") and len(quantizer.codebooks) > layer_idx:
                    codebook = quantizer.codebooks[layer_idx]
                    if hasattr(codebook, "embed"):
                        layer_embeds = codebook.embed(layer_codes)  # [B, T, D]
                    else:
                        # Fallback: use embedding lookup
                        layer_embeds = codebook[layer_codes]  # [B, T, D]
                elif hasattr(quantizer, "embed"):
                    # Single codebook for all layers
                    layer_embeds = quantizer.embed(layer_codes)  # [B, T, D]
                else:
                    raise ValueError("Could not access quantizer embeddings")
                
                embeddings_list.append(layer_embeds)
            
            # Average embeddings across layers
            embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)  # [B, T, D]
            
            return embeddings, speech_lens
    
    def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute output lengths for SpeechTokenizer.
        SpeechTokenizer stride is typically 320 samples.
        """
        return (input_lengths / 320).ceil().long()
