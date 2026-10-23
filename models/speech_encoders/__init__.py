"""Speech encoders used by the AudioLLM pipeline (XLS-R / Wav2Vec2)."""

from .base import BaseSpeechEncoder
from .wav2vec2_encoder import Wav2Vec2Encoder


def create_speech_encoder(
    encoder_type, encoder_path, encoder_source="fairseq", **kwargs
):
    encoder_type = encoder_type.lower()
    kwargs = dict(kwargs)

    if encoder_type in ("wav2vec2", "wav2vec2-bert", "xlsr", "xls-r"):
        kwargs["encoder_source"] = encoder_source
        if encoder_type == "wav2vec2-bert":
            kwargs["use_bert"] = True
        return Wav2Vec2Encoder(encoder_path, **kwargs)

    raise ValueError(
        f"Unsupported encoder type '{encoder_type}'. "
        "Supported encoders: wav2vec2, wav2vec2-bert, xlsr."
    )


__all__ = [
    "BaseSpeechEncoder",
    "Wav2Vec2Encoder",
    "create_speech_encoder",
]
