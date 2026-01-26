"""
Speech Encoders Package

This package provides a unified interface for various speech/audio encoders.
All encoders inherit from BaseSpeechEncoder and implement the same interface.
"""

from .base import BaseSpeechEncoder

# Import encoders (lazy loading to avoid import errors)
try:
    from .whisper_encoder import WhisperEncoder
except ImportError:
    WhisperEncoder = None

try:
    from .wav2vec2_encoder import Wav2Vec2Encoder
except ImportError:
    Wav2Vec2Encoder = None

try:
    from .wavlm_encoder import WavLMEncoder
except ImportError:
    WavLMEncoder = None

try:
    from .speechtokenizer_encoder import SpeechTokenizerEncoder
except ImportError:
    SpeechTokenizerEncoder = None

try:
    from .sqcodec_encoder import SQCodecEncoder
except ImportError:
    SQCodecEncoder = None

try:
    from .eat_encoder import EATEncoder
except ImportError:
    EATEncoder = None

__all__ = [
    "BaseSpeechEncoder",
    "WhisperEncoder",
    "Wav2Vec2Encoder",
    "WavLMEncoder",
    "SpeechTokenizerEncoder",
    "SQCodecEncoder",
    "EATEncoder",
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
    
    if encoder_type == "whisper":
        return WhisperEncoder(encoder_path, **kwargs)
    elif encoder_type == "whisper_ner":
        return WhisperEncoder(encoder_path, use_ner=True, **kwargs)
    elif encoder_type == "wav2vec2":
        return Wav2Vec2Encoder(encoder_path, encoder_source=encoder_source, **kwargs)
    elif encoder_type == "wav2vec2-bert":
        return Wav2Vec2Encoder(encoder_path, encoder_source="huggingface", use_bert=True, **kwargs)
    elif encoder_type == "wavlm":
        return WavLMEncoder(encoder_path, encoder_source=encoder_source, **kwargs)
    elif encoder_type == "speechtokenizer":
        return SpeechTokenizerEncoder(encoder_path, **kwargs)
    elif encoder_type == "sqcodec":
        return SQCodecEncoder(encoder_path, **kwargs)
    elif encoder_type == "eat":
        return EATEncoder(encoder_path, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
