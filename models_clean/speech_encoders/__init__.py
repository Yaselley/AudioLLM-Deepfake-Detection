"""
Speech Encoders Package

This package provides a unified interface for various speech/audio encoders.
All encoders inherit from BaseSpeechEncoder and implement the same interface.
"""

from .base import BaseSpeechEncoder

# Import encoders (lazy loading to avoid import errors)
from .whisper_encoder import WhisperEncoder
from .wav2vec2_encoder import Wav2Vec2Encoder
from .wavlm_encoder import WavLMEncoder
from .speechtokenizer_encoder import SpeechTokenizerEncoder
from .sqcodec_encoder import SQCodecEncoder
from .eat_encoder import EATEncoder
from .dac_encoder import DACEncoder

__all__ = [
    "BaseSpeechEncoder",
    "WhisperEncoder",
    "Wav2Vec2Encoder",
    "WavLMEncoder",
    "SpeechTokenizerEncoder",
    "SQCodecEncoder",
    "EATEncoder",
    "DACEncoder",
]

def create_speech_encoder(encoder_type, encoder_path, encoder_source="huggingface", **kwargs):
    """
    Factory function to create a speech encoder.
    
    Args:
        encoder_type: Type of encoder ("whisper", "wav2vec2", "wavlm", "speechtokenizer", "sqcodec", "eat")
        encoder_path: Path to the encoder model/checkpoint
        encoder_source: Source of the encoder ("huggingface", "fairseq", "unil", etc.)
        **kwargs: Additional arguments specific to each encoder type
    
    Returns:
        BaseSpeechEncoder instance
    """
    encoder_type = encoder_type.lower()
    # Normalize kwargs so we can safely override keys without
    # accidentally passing duplicates to constructors.
    kwargs = dict(kwargs)
    
    if encoder_type == "whisper":
        return WhisperEncoder(encoder_path, **kwargs)
    elif encoder_type == "whisper_ner":
        # Ensure NER variant is enabled without passing duplicate kwargs.
        kwargs["use_ner"] = True
        return WhisperEncoder(encoder_path, **kwargs)
    elif encoder_type == "wav2vec2":
        kwargs["encoder_source"] = encoder_source
        return Wav2Vec2Encoder(encoder_path, **kwargs)
    elif encoder_type == "wav2vec2-bert":
        kwargs["encoder_source"] = "huggingface"
        kwargs["use_bert"] = True
        return Wav2Vec2Encoder(encoder_path, **kwargs)
    elif encoder_type == "wavlm":
        kwargs["encoder_source"] = encoder_source
        return WavLMEncoder(encoder_path, **kwargs)
    elif encoder_type == "speechtokenizer":
        return SpeechTokenizerEncoder(encoder_path, **kwargs)
    elif encoder_type == "sqcodec":
        return SQCodecEncoder(encoder_path, **kwargs)
    elif encoder_type == "eat":
        kwargs["encoder_source"] = encoder_source
        return EATEncoder(encoder_path, **kwargs)
    elif encoder_type == "dac":
        return DACEncoder(encoder_path, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
