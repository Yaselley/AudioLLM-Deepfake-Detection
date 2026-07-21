"""
Speech encoders used by the AudioLLM pipeline (Whisper / Wav2Vec2 / EAT + OpenSmile).
"""

from .base import BaseSpeechEncoder
from .whisper_encoder import WhisperEncoder
from .wav2vec2_encoder import Wav2Vec2Encoder
from .eat_encoder import EATEncoder


def create_speech_encoder(encoder_type, encoder_path, encoder_source="huggingface", **kwargs):
    encoder_type = encoder_type.lower()
    kwargs = dict(kwargs)

    if encoder_type in ("whisper", "whisper_ner"):
        if encoder_type == "whisper_ner":
            kwargs["use_ner"] = True
        return WhisperEncoder(encoder_path, **kwargs)
    if encoder_type in ("wav2vec2", "wav2vec2-bert"):
        kwargs["encoder_source"] = encoder_source
        if encoder_type == "wav2vec2-bert":
            kwargs["use_bert"] = True
        return Wav2Vec2Encoder(encoder_path, **kwargs)
    if encoder_type == "eat":
        kwargs["encoder_source"] = encoder_source
        return EATEncoder(encoder_path, **kwargs)

    raise ValueError(
        f"Unsupported encoder type '{encoder_type}'. "
        "Supported encoders: whisper, wav2vec2, eat. Other encoders are archived under "
        "/ds-slt/audio/yelkheir/AudioLLM-Deepfake-Detection-legacy/models/."
    )


__all__ = [
    "BaseSpeechEncoder",
    "WhisperEncoder",
    "Wav2Vec2Encoder",
    "EATEncoder",
    "create_speech_encoder",
]
