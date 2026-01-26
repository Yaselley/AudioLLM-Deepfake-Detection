# """
# DAC (Descript Audio Codec) encoder implementation.
# Supports both discrete tokens and continuous embeddings.
# Operates at 16kHz sampling rate.
# """

# import os
# import logging
# import torch
# import torch.nn as nn
# from typing import Tuple, Optional

# from .base import BaseSpeechEncoder


# class DACEncoder(BaseSpeechEncoder):
#     """
#     DAC (Descript Audio Codec) encoder.
    
#     Can return either:
#     1. Discrete tokens: [B, n_q, T] discrete code values
#     2. Continuous embeddings: [B, T, D] continuous embeddings from quantizer
    
#     DAC operates at 16kHz sampling rate by default.
#     """
    
#     def __init__(
#         self,
#         encoder_path: str,
#         return_discrete_tokens: bool = True,
#         num_codebooks: Optional[int] = None,
#         codebook_size: Optional[int] = None,
#         model_type: str = "16khz",  # "16khz", "24khz", "44khz"
#         freeze: bool = True
#     ):
#         """
#         Args:
#             encoder_path: Path to DAC model checkpoint or model type
#             return_discrete_tokens: If True, return discrete codes [B, n_q, T]
#                                    If False, return continuous embeddings [B, T, D]
#             num_codebooks: Number of codebooks (auto-detected if None)
#             codebook_size: Codebook size (auto-detected if None)
#             model_type: DAC model type - "16khz" (default), "24khz", or "44khz"
#             freeze: Whether to freeze encoder parameters (always True for DAC)
#         """
#         super().__init__(freeze=True)  # DAC is always frozen
        
#         # Load DAC
#         try:
#             import dac
#         except ImportError:
#             raise ImportError("DAC not found. Please install: pip install descript-audio-codec")
        
#         # Load model
#         logging.info(f"Loading DAC model: {model_type}")
        
#         # Load pretrained model by type
#         model_path = dac.utils.download(model_type="16khz")            
#         self.encoder = dac.DAC.load(model_path)
        
#         self.encoder.eval()
#         for param in self.encoder.parameters():
#             param.requires_grad = False
        
#         # Get codebook info from the loaded model
#         # DAC stores n_codebooks and codebook_size in the quantizer
#         if hasattr(self.encoder, "quantizer"):
#             if hasattr(self.encoder.quantizer, "n_codebooks"):
#                 detected_num_codebooks = self.encoder.quantizer.n_codebooks
#             elif hasattr(self.encoder.quantizer, "quantizers"):
#                 detected_num_codebooks = len(self.encoder.quantizer.quantizers)
#             else:
#                 detected_num_codebooks = None
            
#             if hasattr(self.encoder.quantizer, "codebook_size"):
#                 detected_codebook_size = self.encoder.quantizer.codebook_size
#             else:
#                 detected_codebook_size = None
#         else:
#             detected_num_codebooks = None
#             detected_codebook_size = None
        
#         # Use detected values or fall back to user-provided values
#         if num_codebooks is not None:
#             self.num_codebooks = num_codebooks
#             logging.info(f"Using user-specified num_codebooks: {self.num_codebooks}")
#         elif detected_num_codebooks is not None:
#             self.num_codebooks = detected_num_codebooks
#             logging.info(f"Auto-detected num_codebooks from model: {self.num_codebooks}")
#         else:
#             # Default for 16kHz DAC
#             self.num_codebooks = 9
#             logging.warning(f"Could not detect num_codebooks, using default: {self.num_codebooks}")
        
#         if codebook_size is not None:
#             self.codebook_size = codebook_size
#             logging.info(f"Using user-specified codebook_size: {self.codebook_size}")
#         elif detected_codebook_size is not None:
#             self.codebook_size = detected_codebook_size
#             logging.info(f"Auto-detected codebook_size from model: {self.codebook_size}")
#         else:
#             # Default for DAC
#             self.codebook_size = 1024
#             logging.warning(f"Could not detect codebook_size, using default: {self.codebook_size}")
        
#         logging.info(f"DAC: {self.num_codebooks} codebooks, codebook size {self.codebook_size}")
        
#         # Get sample rate and hop length
#         self.sample_rate = getattr(self.encoder, "sample_rate", 16000)
#         # DAC 16kHz typically has hop_length of 320 (downsampling factor)
#         if hasattr(self.encoder, "hop_length"):
#             self.hop_length = self.encoder.hop_length
#         elif hasattr(self.encoder.encoder, "hop_length"):
#             self.hop_length = self.encoder.encoder.hop_length
#         else:
#             # Calculate from encoder blocks
#             self.hop_length = 320  # Default for 16kHz
#             logging.warning(f"Could not detect hop_length, using default: {self.hop_length}")
        
#         logging.info(f"DAC sample rate: {self.sample_rate}Hz, hop length: {self.hop_length}")
        
#         # Set output mode
#         self.return_discrete_tokens = return_discrete_tokens
        
#         # Set is_discrete property
#         self._is_discrete = return_discrete_tokens
        
#         if return_discrete_tokens:
#             # Discrete mode: output_dim is None
#             self._output_dim = None
#         else:
#             # Continuous mode: get embedding dimension from encoder
#             # DAC encoder typically outputs 1024-dimensional embeddings
#             if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "channels"):
#                 # channels gives the embedding dimension
#                 self._output_dim = self.encoder.encoder.channels
#             elif hasattr(self.encoder, "latent_dim"):
#                 self._output_dim = self.encoder.latent_dim
#             else:
#                 # Default fallback
#                 self._output_dim = 1024
#                 logging.warning(f"Could not determine embedding dimension, using default: {self._output_dim}")
    
#     @property
#     def is_discrete(self) -> bool:
#         """Return whether this encoder outputs discrete tokens."""
#         return self._is_discrete
    
#     @property
#     def output_dim(self) -> Optional[int]:
#         """Return output dimension (None for discrete encoders)."""
#         return self._output_dim
    
#     def encode(self, raw_wav: torch.Tensor, audio_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Encode audio using DAC.
        
#         Args:
#             raw_wav: [B, T] raw audio at 16kHz
#             audio_attention_mask: [B, T] attention mask
        
#         Returns:
#             If return_discrete_tokens=True:
#                 codes: [B, n_q, T] discrete code values
#             If return_discrete_tokens=False:
#                 embeddings: [B, T, D] continuous embeddings (post-quantization)
#             lengths: [B] valid lengths
#         """
#         device = raw_wav.device
        
#         # Calculate input lengths
#         if audio_attention_mask is not None:
#             input_lengths = audio_attention_mask.sum(dim=1)
#         else:
#             input_lengths = torch.tensor([raw_wav.shape[1]] * raw_wav.shape[0], device=device)
        
#         print(raw_wav.shape)
#         # Prepare input
#         with torch.no_grad():
#             if raw_wav.dtype != torch.float32:
#                 raw_wav_input = raw_wav.to(torch.float32)
#             else:
#                 raw_wav_input = raw_wav
            
#             # DAC expects [B, 1, T]
#             wav_input = raw_wav_input.unsqueeze(1)
            
#             if self.return_discrete_tokens:
#                 # Discrete mode: use encode to get codes
#                 # DAC encode returns a tuple (z, codes, latents, commitment_loss, codebook_loss)
#                 # codes shape: [B, n_q, T]
#                 output = self.encoder.encode(wav_input, n_quantizers=self.num_codebooks)
#                 codes = output[1]  # [B, n_q, T]
                
#                 # Calculate output lengths
#                 batch_size = codes.shape[0]
#                 max_code_len = codes.shape[2]
                
#                 # DAC hop_length for downsampling
#                 speech_lens = (input_lengths / self.hop_length).ceil().long()
#                 speech_lens = torch.clamp(speech_lens, max=max_code_len)
#                 print(codes.shape)

#                 return codes, speech_lens
#             else:
#                 # Continuous mode: get post-quantization embeddings
#                 # First encode to get the latent representation
#                 output = self.encoder.encode(wav_input, n_quantizers=self.num_codebooks)
#                 z = output[0]  # Latent representation [B, D, T]
#                 codes = output[1]  # Codes [B, n_q, T]
                
#                 # Option 1: Use the pre-quantization latent z
#                 # embeddings = z.transpose(1, 2)  # [B, T, D]
                
#                 # Option 2: Use quantized embeddings (sum across codebooks)
#                 # This is more consistent with SpeechTokenizer approach
#                 quantized_embeddings = []
#                 for i in range(self.num_codebooks):
#                     # Get codebook for this layer
#                     codebook_i = self.encoder.quantizer.quantizers[i]
#                     codes_i = codes[:, i, :]  # [B, T]
                    
#                     # Lookup embeddings
#                     if hasattr(codebook_i, "codebook"):
#                         emb_i = torch.nn.functional.embedding(codes_i, codebook_i.codebook)
#                     elif hasattr(codebook_i, "embed"):
#                         emb_i = codebook_i.embed(codes_i)
#                     else:
#                         # Fallback: use the pre-quantization latent
#                         logging.warning(f"Could not access codebook {i}, using pre-quantization latent")
#                         embeddings = z.transpose(1, 2)  # [B, T, D]
#                         break
                    
#                     quantized_embeddings.append(emb_i)
#                 else:
#                     # Sum across codebooks (residual VQ structure)
#                     embeddings = torch.stack(quantized_embeddings, dim=0).sum(dim=0)  # [B, T, D]
                
#                 # Calculate output lengths
#                 batch_size = embeddings.shape[0]
#                 max_emb_len = embeddings.shape[1]
                
#                 speech_lens = (input_lengths / self.hop_length).ceil().long()
#                 speech_lens = torch.clamp(speech_lens, max=max_emb_len)
                
#                 return embeddings, speech_lens
    
#     def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
#         """
#         Compute output lengths for DAC.
#         DAC hop_length is typically 320 for 16kHz model.
#         """
#         return (input_lengths / 320).floor().long()

"""
DAC (Descript Audio Codec) encoder implementation.
Supports both discrete tokens and continuous embeddings.
Operates at 16kHz sampling rate.
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .base import BaseSpeechEncoder


class DACEncoder(BaseSpeechEncoder):
    """
    DAC (Descript Audio Codec) encoder.
    
    Can return either:
    1. Discrete tokens: [B, n_q, T] discrete code values
    2. Continuous embeddings: [B, T, D] continuous embeddings from quantizer
    
    DAC operates at 16kHz sampling rate by default.
    """
    
    def __init__(
        self,
        encoder_path: str,
        return_discrete_tokens: bool = True,
        num_codebooks: Optional[int] = None,
        codebook_size: Optional[int] = None,
        model_type: str = "16khz",  # "16khz", "24khz", "44khz"
        freeze: bool = True
    ):
        """
        Args:
            encoder_path: Path to DAC model checkpoint or model type
            return_discrete_tokens: If True, return discrete codes [B, n_q, T]
                                   If False, return continuous embeddings [B, T, D]
            num_codebooks: Number of codebooks (auto-detected if None)
            codebook_size: Codebook size (auto-detected if None)
            model_type: DAC model type - "16khz" (default), "24khz", or "44khz"
            freeze: Whether to freeze encoder parameters (always True for DAC)
        """
        super().__init__(freeze=True)  # DAC is always frozen
        
        # Load DAC
        try:
            import dac
        except ImportError:
            raise ImportError("DAC not found. Please install: pip install descript-audio-codec")
        
        # Load model
        logging.info(f"Loading DAC model: {model_type}")
        
        model_path = dac.utils.download(model_type="16khz")
        self.encoder = dac.DAC.load(model_path)
        
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Get codebook info from the loaded model
        # DAC stores n_codebooks and codebook_size in the quantizer
        if hasattr(self.encoder, "quantizer"):
            if hasattr(self.encoder.quantizer, "n_codebooks"):
                detected_num_codebooks = self.encoder.quantizer.n_codebooks
            elif hasattr(self.encoder.quantizer, "quantizers"):
                detected_num_codebooks = len(self.encoder.quantizer.quantizers)
            else:
                detected_num_codebooks = None
            
            if hasattr(self.encoder.quantizer, "codebook_size"):
                detected_codebook_size = self.encoder.quantizer.codebook_size
            else:
                detected_codebook_size = None
        else:
            detected_num_codebooks = None
            detected_codebook_size = None
        
        # Use detected values or fall back to user-provided values
        if num_codebooks is not None:
            self.num_codebooks = num_codebooks
            logging.info(f"Using user-specified num_codebooks: {self.num_codebooks}")
        elif detected_num_codebooks is not None:
            self.num_codebooks = detected_num_codebooks
            logging.info(f"Auto-detected num_codebooks from model: {self.num_codebooks}")
        else:
            # Default for 16kHz DAC is 12 codebooks
            # 24kHz and 44kHz use 32 codebooks
            if model_type == "16khz":
                self.num_codebooks = 12
            else:  # 24khz or 44khz
                self.num_codebooks = 32
            logging.warning(f"Could not detect num_codebooks, using default for {model_type}: {self.num_codebooks}")
        
        if codebook_size is not None:
            self.codebook_size = codebook_size
            logging.info(f"Using user-specified codebook_size: {self.codebook_size}")
        elif detected_codebook_size is not None:
            self.codebook_size = detected_codebook_size
            logging.info(f"Auto-detected codebook_size from model: {self.codebook_size}")
        else:
            # Default for DAC
            self.codebook_size = 1024
            logging.warning(f"Could not detect codebook_size, using default: {self.codebook_size}")
        
        logging.info(f"DAC: {self.num_codebooks} codebooks, codebook size {self.codebook_size}")
        
        # Get sample rate and hop length
        self.sample_rate = getattr(self.encoder, "sample_rate", 16000)
        # DAC 16kHz typically has hop_length of 320 (downsampling factor)
        if hasattr(self.encoder, "hop_length"):
            self.hop_length = self.encoder.hop_length
        elif hasattr(self.encoder.encoder, "hop_length"):
            self.hop_length = self.encoder.encoder.hop_length
        else:
            # Calculate from encoder blocks
            self.hop_length = 320  # Default for 16kHz
            logging.warning(f"Could not detect hop_length, using default: {self.hop_length}")
        
        logging.info(f"DAC sample rate: {self.sample_rate}Hz, hop length: {self.hop_length}")
        
        # Set output mode
        self.return_discrete_tokens = return_discrete_tokens
        
        # Set is_discrete property
        self._is_discrete = return_discrete_tokens
        
        if return_discrete_tokens:
            # Discrete mode: output_dim is None
            self._output_dim = None
        else:
            # Continuous mode: get embedding dimension from encoder
            # DAC encoder typically outputs 1024-dimensional embeddings
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "channels"):
                # channels gives the embedding dimension
                self._output_dim = self.encoder.encoder.channels
            elif hasattr(self.encoder, "latent_dim"):
                self._output_dim = self.encoder.latent_dim
            else:
                # Default fallback
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
        Encode audio using DAC.
        
        Args:
            raw_wav: [B, T] raw audio at 16kHz
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
            
            # DAC expects [B, 1, T]
            wav_input = raw_wav_input.unsqueeze(1)
            
            if self.return_discrete_tokens:
                # Discrete mode: use encode to get codes
                # DAC encode returns a tuple (z, codes, latents, commitment_loss, codebook_loss)
                # codes shape: [B, n_q, T]
                output = self.encoder.encode(wav_input, n_quantizers=self.num_codebooks)
                codes = output[1]  # [B, n_q, T]
                
                # Calculate output lengths
                batch_size = codes.shape[0]
                max_code_len = codes.shape[2]
                
                # DAC hop_length for downsampling
                speech_lens = (input_lengths / self.hop_length).ceil().long()
                speech_lens = torch.clamp(speech_lens, max=max_code_len)
                
                return codes, speech_lens
            else:
                # Continuous mode: use pre-quantization latent
                # The latent z from encoder is [B, D, T] where D=1024
                # This is the most reliable way to get continuous embeddings
                output = self.encoder.encode(wav_input, n_quantizers=self.num_codebooks)
                z = output[0]  # Latent representation [B, D, T] where D=1024
                
                # Transpose to [B, T, D]
                embeddings = z.transpose(1, 2)  # [B, T, 1024]
                
                # Calculate output lengths
                batch_size = embeddings.shape[0]
                max_emb_len = embeddings.shape[1]
                
                speech_lens = (input_lengths / self.hop_length).ceil().long()
                speech_lens = torch.clamp(speech_lens, max=max_emb_len)
                
                return embeddings, speech_lens
    
    def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute output lengths for DAC.
        DAC hop_length is typically 320 for 16kHz model.
        """
        return (input_lengths / self.hop_length).ceil().long()