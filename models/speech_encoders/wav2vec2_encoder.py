# """
# Wav2Vec2 encoder implementation.
# """

# import logging
# import torch
# from typing import Tuple, Optional

# from transformers import Wav2Vec2Model, Wav2Vec2BertModel, AutoFeatureExtractor
# from .base import BaseSpeechEncoder


# class Wav2Vec2Encoder(BaseSpeechEncoder):
#     """Wav2Vec2 encoder for continuous speech embeddings."""
    
#     def __init__(
#         self,
#         encoder_path: str,
#         encoder_source: str = "huggingface",
#         use_bert: bool = False,
#         use_weighted_layer_sum: bool = False,
#         weighted_layer_norm: str = "none",
#         freeze: bool = True
#     ):
#         super().__init__(freeze=freeze)
        
#         self.encoder_source = encoder_source.lower()
#         self.use_bert = use_bert
#         self.use_weighted_layer_sum = use_weighted_layer_sum
#         self.weighted_layer_norm = weighted_layer_norm
#         self.layer_pooler = None
        
#         if use_bert:
#             self.encoder = Wav2Vec2BertModel.from_pretrained(encoder_path)
#             self._output_dim = self.encoder.config.hidden_size
#             self.processor = AutoFeatureExtractor.from_pretrained(encoder_path)
#         elif self.encoder_source == "fairseq":
#             import fairseq
#             models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([encoder_path])
#             self.encoder = models[0]
#             self._output_dim = cfg.model.encoder_embed_dim
#         else:
#             self.encoder = Wav2Vec2Model.from_pretrained(encoder_path)
#             self._output_dim = self.encoder.config.hidden_size

#         # Optional weighted layer pooling (HF only)
#         if self.use_weighted_layer_sum:
#             if self.encoder_source == "fairseq":
#                 raise ValueError("Weighted layer pooling is only supported for HuggingFace wav2vec2 models.")
#             from .weighted_layer_pooling import WeightedLayerPooling
#             num_layers = 24
#             self.layer_pooler = WeightedLayerPooling(num_layers=num_layers, norm=self.weighted_layer_norm)
        
#         if freeze:
#             self.freeze_encoder()
    
#     def encode(self, raw_wav: torch.Tensor, audio_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         device = raw_wav.device
        
#         if audio_attention_mask is not None:
#             input_lengths = audio_attention_mask.sum(dim=1)
#         else:
#             input_lengths = torch.tensor([raw_wav.shape[1]] * raw_wav.shape[0], device=device)
        
#         if self.encoder_source == "fairseq":
#             if raw_wav.dtype != torch.float32:
#                 raw_wav_input = raw_wav.to(torch.float32)
#             else:
#                 raw_wav_input = raw_wav
#             res = self.encoder(raw_wav_input, mask=False, features_only=True)
#             hidden_states = res["x"]
#         elif self.use_bert:
#             with torch.no_grad():
#                 raw_wav_cpu = raw_wav.cpu().numpy()
#                 inputs = self.processor([w for w in raw_wav_cpu], sampling_rate=16000, return_tensors="pt", padding=True)
#                 input_values = inputs.input_values.to(device)
#             outputs = self.encoder(input_values, output_hidden_states=self.use_weighted_layer_sum, return_dict=True)
#             if self.use_weighted_layer_sum and self.layer_pooler is not None:
#                 hidden_states = self.layer_pooler(outputs.hidden_states)
#             else:
#                 hidden_states = outputs.last_hidden_state
#         else:
#             outputs = self.encoder(raw_wav, output_hidden_states=self.use_weighted_layer_sum, return_dict=True)
#             if self.use_weighted_layer_sum and self.layer_pooler is not None:
#                 hidden_states = self.layer_pooler(outputs.hidden_states)
#             else:
#                 hidden_states = outputs.last_hidden_state
        
#         lengths = self.get_feat_extract_output_lengths(input_lengths)
#         return hidden_states, lengths
    
#     def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
#         if self.use_bert:
#             feat_lens = (input_lengths / 160).ceil().long()
#             if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
#                 return self.encoder._get_feat_extract_output_lengths(feat_lens)
#             return (feat_lens + 3) // 4
#         elif hasattr(self.encoder, "_get_feat_extract_output_lengths"):
#             return self.encoder._get_feat_extract_output_lengths(input_lengths)
#         else:
#             return (input_lengths / 320).ceil().long()

"""
Wav2Vec2 encoder implementation.
"""

import logging
import torch
from typing import Tuple, Optional

from transformers import Wav2Vec2Model, Wav2Vec2BertModel, AutoFeatureExtractor
from .base import BaseSpeechEncoder


class Wav2Vec2Encoder(BaseSpeechEncoder):
    """Wav2Vec2 encoder for continuous speech embeddings."""
    
    def __init__(
        self,
        encoder_path: str,
        encoder_source: str = "huggingface",
        use_bert: bool = False,
        use_weighted_layer_sum: bool = False,
        weighted_layer_norm: str = "none",
        freeze: bool = True
    ):
        super().__init__(freeze=freeze)
        
        self.encoder_source = encoder_source.lower()
        self.use_bert = use_bert
        self.use_weighted_layer_sum = use_weighted_layer_sum
        self.weighted_layer_norm = weighted_layer_norm
        self.layer_pooler = None
        
        if use_bert:
            self.encoder = Wav2Vec2BertModel.from_pretrained(encoder_path)
            self._output_dim = self.encoder.config.hidden_size
            self.processor = AutoFeatureExtractor.from_pretrained(encoder_path)
        elif self.encoder_source == "fairseq":
            import fairseq
            print(f"DEBUG: Wav2Vec2Encoder loading Fairseq model from {encoder_path}")
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([encoder_path])
            self.encoder = models[0]
            self._output_dim = cfg.model.encoder_embed_dim
            print(f"DEBUG: Loaded Fairseq model with output_dim={self._output_dim}")
        else:
            self.encoder = Wav2Vec2Model.from_pretrained(encoder_path)
            self._output_dim = self.encoder.config.hidden_size

        # Optional weighted layer pooling
        if self.use_weighted_layer_sum:
            from .weighted_layer_pooling import WeightedLayerPooling
            # Determine number of layers based on encoder type
            if self.encoder_source == "fairseq":
                # Fairseq wav2vec2 models typically have 12 or 24 layers
                # Try to infer from encoder config or use default
                if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layers'):
                    num_layers = len(self.encoder.encoder.layers)
                else:
                    # Default to 12 for fairseq models (can be overridden)
                    num_layers = 12
                    logging.warning(f"Could not determine number of layers for fairseq model, defaulting to {num_layers}")
            else:
                num_layers = 24  # HuggingFace default
            
            self.layer_pooler = WeightedLayerPooling(num_layers=num_layers, norm=self.weighted_layer_norm)
        
        if freeze:
            self.freeze_encoder()
            # Ensure layer_pooler remains trainable even when encoder is frozen
            if self.layer_pooler is not None:
                for param in self.layer_pooler.parameters():
                    param.requires_grad = True
    
    def encode(self, raw_wav: torch.Tensor, audio_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = raw_wav.device
        
        if audio_attention_mask is not None:
            input_lengths = audio_attention_mask.sum(dim=1)
        else:
            input_lengths = torch.tensor([raw_wav.shape[1]] * raw_wav.shape[0], device=device)
        
        if self.encoder_source == "fairseq":
            if raw_wav.dtype != torch.float32:
                raw_wav_input = raw_wav.to(torch.float32)
            else:
                raw_wav_input = raw_wav
            
            # Request layer outputs if using weighted pooling
            res = self.encoder(
                raw_wav_input, 
                mask=False, 
                features_only=True,
                layer=None if self.use_weighted_layer_sum else None
            )
            
            if self.use_weighted_layer_sum and self.layer_pooler is not None:
                if "layer_results" in res and res["layer_results"] is not None:
                    layer_outputs = [layer[0].transpose(0, 1) for layer in res["layer_results"]]
                    stacked_layers = torch.stack(layer_outputs, dim=1)
                    hidden_states = self.layer_pooler(stacked_layers)
                else:
                    logging.warning("Layer results not available for weighted pooling, using final output")
                    hidden_states = res["x"]
            else:
                hidden_states = res["x"]


        elif self.use_bert:
            with torch.no_grad():
                raw_wav_cpu = raw_wav.cpu().numpy()
                inputs = self.processor([w for w in raw_wav_cpu], sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
            outputs = self.encoder(input_values, output_hidden_states=self.use_weighted_layer_sum, return_dict=True)
            if self.use_weighted_layer_sum and self.layer_pooler is not None:
                hidden_states = self.layer_pooler(outputs.hidden_states)
            else:
                hidden_states = outputs.last_hidden_state
        else:
            outputs = self.encoder(raw_wav, output_hidden_states=self.use_weighted_layer_sum, return_dict=True)
            if self.use_weighted_layer_sum and self.layer_pooler is not None:
                hidden_states = self.layer_pooler(outputs.hidden_states)
            else:
                hidden_states = outputs.last_hidden_state
        
        lengths = self.get_feat_extract_output_lengths(input_lengths)
        return hidden_states, lengths
    
    def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        if self.use_bert:
            feat_lens = (input_lengths / 160).ceil().long()
            if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
                return self.encoder._get_feat_extract_output_lengths(feat_lens)
            return (feat_lens + 3) // 4
        elif hasattr(self.encoder, "_get_feat_extract_output_lengths"):
            return self.encoder._get_feat_extract_output_lengths(input_lengths)
        else:
            return (input_lengths / 320).ceil().long()