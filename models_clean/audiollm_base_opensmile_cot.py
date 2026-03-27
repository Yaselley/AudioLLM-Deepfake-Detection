"""
AudioLLM Base Model (OpenSMILE CoT) - Model learns to predict OpenSMILE features.

This module provides a Chain-of-Thought (CoT) implementation where the model
predicts the OpenSMILE features as an intermediate step (<ANALYSE> tags)
before predicting the final label.

The OpenSMILE features are NOT provided as input, but used as target ground truth
during training.
"""

import logging
import contextlib
import torch
import torch.nn as nn
import sys
import os

import typing
if not hasattr(typing, 'Self'):
    try:
        from typing_extensions import Self
        typing.Self = Self
    except ImportError:
        class _SelfType:
            def __class_getitem__(cls, item):
                return item
        typing.Self = _SelfType

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append("/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/models_clean")

# Import separated speech encoders
from speech_encoders import create_speech_encoder, BaseSpeechEncoder

# Add DeepFense path for custom modules if needed
sys.path.append("/netscratch/yelkheir/DeepFense/DeepFense")


class AudioDeepfakeLLM(nn.Module):
    """
    Audio-LLM for Deepfake Detection with CoT Analysis.
    
    Training:
    - Input: Audio + Prompt
    - Target: <ANALYSE> {Predicted Features} </ANALYSE> {Label}
    
    Inference:
    - Input: Audio + Prompt
    - Output: <ANALYSE> {Generated Features} </ANALYSE> {Label}
    """
    
    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.amp.autocast('cuda', dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        # LLM Args
        llm_path="",
        llm_type="qwen2",  # qwen2, qwen3, llama
        
        # Audio Encoder Args (using separated encoders)
        audio_encoder_path="",
        audio_encoder_type="speechtokenizer",  # speechtokenizer, sqcodec, whisper, wav2vec2, wavlm, eat
        audio_encoder_source="huggingface",  # huggingface, fairseq, unil
        freeze_audio_encoder=True,

        # Weighted layer pooling (Whisper / Wav2Vec2 only)
        use_weighted_layer_sum: bool = False,
        weighted_layer_norm: str = "none",  # "none" or "batchnorm"
        
        # SpeechTokenizer specific
        speechtokenizer_num_layers=8,
        speechtokenizer_codebook_size=1024,
        speechtokenizer_return_discrete=True,  # True: discrete tokens, False: continuous embeddings
        
        # SQ-Codec specific
        sqcodec_num_quantizers=4,
        sqcodec_num_levels=4096,
        
        # DAC specific
        dac_num_codebooks=1,
        dac_codebook_size=1024,
        dac_model_type="16khz",
        dac_return_discrete=True,

        # Adapter config
        freeze_adapter=False,
        
        # OpenSMILE config
        use_opensmile=False,  # Whether to enable OpenSMILE (used for GROUND TRUTH generation only)
        opensmile_only=False,  # Not relevant for CoT mode, but kept for compat
        opensmile_selected_indices=None, # Indices for OpenSMILE selected features
        
        # LoRA config
        lora=True,
        lora_rank=64,
        lora_alpha=16,
        lora_dropout=0.05,
        
        # Tokenizer config
        special_tokens=None,
        
        # Gen config
        max_txt_len=128,
        end_sym="",
        low_resource=False,
        device_8bit=0,
    ):
        super().__init__()

        self.lora = lora
        self.max_txt_len = max_txt_len
        self.low_resource = low_resource
        self.llm_type = llm_type.lower()
        self.use_opensmile = use_opensmile
        self.opensmile_only = False # CoT requires Audio Encoder + Text
        
        # Set default selected indices if not provided
        if opensmile_selected_indices is None:
            self.opensmile_selected_indices = [1, 18, 19, 20, 21, 39, 45, 51, 58, 60, 67, 77, 78, 82, 85, 86]
        else:
            self.opensmile_selected_indices = opensmile_selected_indices

        # ====================
        # 0. Initialize OpenSMILE (for Ground Truth)
        # ====================
        self.opensmile_extractor = None
        if self.use_opensmile:
            from speech_encoders.opensmile_features_selected import OpenSMILESelectedExtractor
            logging.info(f"Initializing OpenSMILE selected feature extractor (Ground Truth) with indices: {self.opensmile_selected_indices}")
            self.opensmile_extractor = OpenSMILESelectedExtractor(selected_indices=self.opensmile_selected_indices)
        
        # ====================
        # 1. Load Tokenizer & LLM
        # ====================
        logging.info(f'Loading {self.llm_type} Tokenizer & Model from {llm_path}')
        
        if "llama" in self.llm_type:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.end_sym = self.tokenizer.eos_token if not end_sym else end_sym
            model_cls = AutoModelForCausalLM
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
        elif "qwen" in self.llm_type:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.end_sym = "<|endoftext|>" if not end_sym else end_sym
            model_cls = AutoModelForCausalLM
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            raise ValueError(f"Unknown LLM type: {self.llm_type}")

        # ====================
        # 2. Create Speech Encoder (using separated encoders)
        # ====================
        if not audio_encoder_type or audio_encoder_type.lower() == "none":
            raise ValueError("audio_encoder_type must be specified for CoT mode")
        
        logging.info(f'Creating {audio_encoder_type} encoder from {audio_encoder_path}')
        
        encoder_kwargs = {
            "freeze": freeze_audio_encoder
        }
        
        # Add encoder-specific kwargs
        if audio_encoder_type.lower() == "speechtokenizer":
            encoder_kwargs.update({
                "return_discrete_tokens": speechtokenizer_return_discrete,
                "num_layers": speechtokenizer_num_layers,
                "codebook_size": speechtokenizer_codebook_size,
            })
        elif audio_encoder_type.lower() == "sqcodec":
            encoder_kwargs.update({
                "num_quantizers": sqcodec_num_quantizers,
                "num_levels": sqcodec_num_levels,
            })
        elif audio_encoder_type.lower() == "whisper_ner":
            encoder_kwargs["use_ner"] = True

        elif audio_encoder_type.lower() == "dac":
            encoder_kwargs.update({
                "return_discrete_tokens": dac_return_discrete,
                "num_codebooks": dac_num_codebooks,
                "codebook_size": dac_codebook_size,
                "model_type": dac_model_type,
            })

        # Weighted layer pooling (only for Whisper / Wav2Vec2)
        if audio_encoder_type.lower() in {"whisper", "whisper_ner", "wav2vec2", "wav2vec2-bert"}:
            encoder_kwargs.update({
                "use_weighted_layer_sum": use_weighted_layer_sum,
                "weighted_layer_norm": weighted_layer_norm,
            })

            self.audio_encoder: BaseSpeechEncoder = create_speech_encoder(
                encoder_type=audio_encoder_type,
                encoder_path=audio_encoder_path,
                encoder_source=audio_encoder_source,
                **encoder_kwargs
            )
            
            # Determine if we're using discrete tokens
            self.use_discrete_tokens = self.audio_encoder.is_discrete
            
            # Get encoder info for token mapping (if discrete)
            if self.use_discrete_tokens:
                if audio_encoder_type.lower() == "speechtokenizer":
                    self.speechtokenizer_num_layers = speechtokenizer_num_layers
                    self.speechtokenizer_codebook_size = speechtokenizer_codebook_size
                    self.speechtokenizer_token_map = {}
                elif audio_encoder_type.lower() == "sqcodec":
                    self.sqcodec_num_quantizers = sqcodec_num_quantizers
                    self.sqcodec_num_levels = sqcodec_num_levels
                    self.sqcodec_token_map = {}
                elif audio_encoder_type.lower() == "dac":
                    self.dac_num_codebooks = dac_num_codebooks
                    self.dac_codebook_size = dac_codebook_size
                    self.dac_token_map = {}
        
        # ====================
        # 3. Add Special Tokens
        # ====================
        self.added_tokens = False
        all_special_tokens = []
        
        # Add discrete tokens if needed
        if self.use_discrete_tokens and audio_encoder_type.lower() == "speechtokenizer":
            for layer_idx in range(self.speechtokenizer_num_layers):
                for code_idx in range(1, self.speechtokenizer_codebook_size + 1):
                    all_special_tokens.append(f"<audio_{layer_idx}_{code_idx}>")
            logging.info(f"Adding {len(all_special_tokens)} SpeechTokenizer tokens")
            
        elif self.use_discrete_tokens and audio_encoder_type.lower() == "sqcodec":
            for quantizer_idx in range(self.sqcodec_num_quantizers):
                for level_idx in range(self.sqcodec_num_levels):
                    all_special_tokens.append(f"<sqcodec_{quantizer_idx}_{level_idx}>")
            logging.info(f"Adding {len(all_special_tokens)} SQ-Codec tokens")
                
        elif self.use_discrete_tokens and audio_encoder_type.lower() == "dac":
            for codebook_idx in range(self.dac_num_codebooks):
                for code_idx in range(self.dac_codebook_size):
                    all_special_tokens.append(f"<dac_{codebook_idx}_{code_idx}>")
            logging.info(f"Adding {len(all_special_tokens)} DAC tokens")

        # Add User Special Tokens (e.g., <ANALYSE>, </ANALYSE>)
        if special_tokens:
            all_special_tokens.extend(special_tokens)
        
        # Ensure ANALYSE tags are in vocab if not already
        cot_tokens = ["<ANALYSE>", "</ANALYSE>"]
        # for t in cot_tokens:
        #     if t not in all_special_tokens:
        #         all_special_tokens.append(t)
        
        if all_special_tokens:
            num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': all_special_tokens})
            if num_added > 0:
                self.added_tokens = True
                logging.info(f"Added {num_added} special tokens to vocabulary")
        
        # ====================
        # 4. Load LLM
        # ====================
        torch_dtype = torch.float16 if self.low_resource else torch.bfloat16
        
        if self.low_resource:
            self.llm_model = model_cls.from_pretrained(
                llm_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit},
                trust_remote_code=True
            )
        else:
            self.llm_model = model_cls.from_pretrained(
                llm_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
        
        # Resize embeddings if tokens added
        if self.added_tokens:
            self.llm_model.resize_token_embeddings(len(self.tokenizer))
        
        # Build token mappings (for discrete encoders)
        if self.use_discrete_tokens and audio_encoder_type.lower() == "speechtokenizer":
            self.speechtokenizer_token_map = {}
            for layer_idx in range(self.speechtokenizer_num_layers):
                for code_idx in range(1, self.speechtokenizer_codebook_size + 1):
                    token_str = f"<audio_{layer_idx}_{code_idx}>"
                    token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                    self.speechtokenizer_token_map[(layer_idx, code_idx - 1)] = token_id
            
        elif self.use_discrete_tokens and audio_encoder_type.lower() == "sqcodec":
            self.sqcodec_token_map = {}
            for quantizer_idx in range(self.sqcodec_num_quantizers):
                for level_idx in range(self.sqcodec_num_levels):
                    token_str = f"<sqcodec_{quantizer_idx}_{level_idx}>"
                    token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                    self.sqcodec_token_map[(quantizer_idx, level_idx)] = token_id
        
        elif self.use_discrete_tokens and audio_encoder_type.lower() == "dac":
            self.dac_token_map = {}
            for codebook_idx in range(self.dac_num_codebooks):
                for code_idx in range(self.dac_codebook_size):
                    token_str = f"<dac_{codebook_idx}_{code_idx}>"
                    token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                    self.dac_token_map[(codebook_idx, code_idx)] = token_id

        # ====================
        # 5. Setup LoRA
        # ====================
        # Freeze LLM
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        if self.lora:
            # For discrete tokenizers, we MUST train embeddings/head to learn the new tokens
            modules_to_save = []
            if self.added_tokens or self.use_discrete_tokens:
                logging.info("Enabling training of embeddings and lm_head for new tokens via modules_to_save")
                modules_to_save = ["embed_tokens", "lm_head"]
            
            # If using SpeechTokenizer with discrete tokens, also enable full training
            if self.use_discrete_tokens and audio_encoder_type.lower() == "speechtokenizer":
                if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "embed_tokens"):
                    self.llm_model.model.embed_tokens.weight.requires_grad = True
                elif hasattr(self.llm_model, "base_model") and hasattr(self.llm_model.base_model, "model"):
                    self.llm_model.base_model.model.embed_tokens.weight.requires_grad = True
                
                if hasattr(self.llm_model, "lm_head"):
                    self.llm_model.lm_head.weight.requires_grad = True
                elif hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "lm_head"):
                    self.llm_model.model.lm_head.weight.requires_grad = True

            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save
            )
            self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            self.llm_model.print_trainable_parameters()

        # ====================
        # 6. Audio Projector (only for continuous embeddings)
        # ====================
        if not self.use_discrete_tokens:
            audio_dim = self.audio_encoder.output_dim
            if audio_dim is None:
                raise ValueError(f"Encoder {audio_encoder_type} returned None for output_dim but is not discrete")
            
            self.speech_proj = nn.Linear(audio_dim, self.llm_model.config.hidden_size)
            if freeze_adapter:
                for param in self.speech_proj.parameters():
                    param.requires_grad = False
                self.speech_proj.eval()
        else:
            self.speech_proj = None

    def encode_speech(self, raw_wav: torch.Tensor, audio_attention_mask=None):
        """
        Encode speech using the separated encoder.
        """
        # Use the separated encoder
        output, lengths = self.audio_encoder.encode(raw_wav, audio_attention_mask)
        
        # Project continuous embeddings to LLM space
        if not self.use_discrete_tokens:
            with self.maybe_autocast():
                output = self.speech_proj(output)
        
        return output, lengths
    
    def extract_opensmile_text(self, raw_wav: torch.Tensor, audio_attention_mask=None):
        """
        Extract OpenSMILE textual features from audio (for Ground Truth).
        """
        if self.opensmile_extractor is None:
            return None
        
        batch_size = raw_wav.shape[0]
        opensmile_texts = []
        
        for i in range(batch_size):
            # Get audio array for this sample
            audio_array = raw_wav[i].cpu().numpy()
            
            # Apply attention mask if provided
            if audio_attention_mask is not None:
                valid_length = int(audio_attention_mask[i].sum().item())
                audio_array = audio_array[:valid_length]
            
            # Extract OpenSMILE features and convert to text
            try:
                text_description = self.opensmile_extractor.process_array(
                    audio_array, 
                    sample_rate=16000, 
                    precision=3
                )
                opensmile_texts.append(text_description)
            except Exception as e:
                logging.warning(f"Error extracting OpenSMILE features for sample {i}: {e}")
                opensmile_texts.append("No analysis available.") 
        
        return opensmile_texts

    def _get_text_embeds(self, token_ids):
        """Helper to get embeddings depending on model structure."""
        embed_module = None
        try:
            embed_module = self.llm_model.get_input_embeddings()
        except:
            if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "embed_tokens"):
                embed_module = self.llm_model.model.embed_tokens
            elif hasattr(self.llm_model, "base_model") and hasattr(self.llm_model.base_model, "model"):
                embed_module = self.llm_model.base_model.model.embed_tokens
            try:
                embed_module = self.llm_model.model.model.embed_tokens
            except:
                embed_module = self.llm_model.embed_tokens
        
        embeds = embed_module(token_ids)
        return embeds.to(embed_module.weight.dtype)

    def prompt_wrap(self, speech_output, speech_lens, prompt, multi_prompt=False, padding_side="right", prefix_embeds=None):
        """
        Wrap speech embeddings/tokens with text prompts.
        Note: OpenSMILE text is NOT injected here for CoT mode.
        """
        if self.use_discrete_tokens:
            return self._prompt_wrap_tokens(speech_output, speech_lens, prompt, multi_prompt, padding_side, prefix_embeds)
        else:
            return self._prompt_wrap_embeds(speech_output, speech_lens, prompt, multi_prompt, padding_side, prefix_embeds)

    def _prompt_wrap_tokens(self, speech_codes, speech_lens, prompt, multi_prompt=False, padding_side="right", prefix_embeds=None):
        """
        Wrap discrete codes with text prompts.
        """
        batch_size = speech_codes.shape[0]
        device = speech_codes.device
        
        # Determine number of layers
        if hasattr(self, 'speechtokenizer_num_layers'):
            n_layers = self.speechtokenizer_num_layers
            token_map = self.speechtokenizer_token_map
        elif hasattr(self, 'sqcodec_num_quantizers'):
            n_layers = self.sqcodec_num_quantizers
            token_map = self.sqcodec_token_map
        elif hasattr(self, 'dac_num_codebooks'):
            n_layers = self.dac_num_codebooks
            token_map = self.dac_token_map
        else:
            n_layers = speech_codes.shape[1]
            token_map = getattr(self, 'speechtokenizer_token_map', getattr(self, 'sqcodec_token_map', {}))
        
        input_ids_list = []
        attn_mask_list = []
        
        if isinstance(prompts := prompt, str):
            prompts = [prompts] * batch_size

        for i, prompt in enumerate(prompts):
            p_before = ""
            p_after = ""
            
            # Clean prompt of <OPENSMILE> tag since we don't use it as input
            clean_prompt = prompt.replace("<OPENSMILE>", "")
            
            # Apply chat template logic (simplified)
            # ... (Assume standard handling of <SpeechHere>) ...
            if "<SpeechHere>" not in clean_prompt:
                p_before, p_after = clean_prompt, ""
            else:
                p_before, p_after = clean_prompt.split("<SpeechHere>")
            
            # Tokenize text parts
            tokens_before = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            tokens_after = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)
            
            # Convert discrete codes to token IDs (same as before)
            valid_len = speech_lens[i].item()
            actual_code_len = speech_codes.shape[2]
            valid_len = min(valid_len, actual_code_len)
            
            cur_codes = speech_codes[i, :, :valid_len]
            
            audio_token_ids = []
            for t in range(valid_len):
                for layer_idx in range(n_layers):
                    code_value = cur_codes[layer_idx, t].item()
                    
                    if hasattr(self, 'speechtokenizer_codebook_size'):
                        token_id = token_map.get((layer_idx, code_value))
                    elif hasattr(self, 'sqcodec_num_levels'):
                        token_id = token_map.get((layer_idx, code_value))
                    elif hasattr(self, 'dac_num_codebooks'):
                        token_id = token_map.get((layer_idx, code_value))
                    else:
                        token_id = token_map.get((layer_idx, code_value))
                    
                    if token_id is not None:
                        audio_token_ids.append(token_id)
            
            audio_token_ids = torch.tensor(audio_token_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            combined_ids = torch.cat([tokens_before.input_ids, audio_token_ids, tokens_after.input_ids], dim=1)
            combined_mask = torch.cat([tokens_before.attention_mask, torch.ones(audio_token_ids.shape, device=device), tokens_after.attention_mask], dim=1)
            
            input_ids_list.append(combined_ids)
            attn_mask_list.append(combined_mask)
        
        # Padding (standard)
        max_len = max([ids.shape[1] for ids in input_ids_list])
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        padded_ids = []
        padded_masks = []
        
        for i in range(batch_size):
            curr_len = input_ids_list[i].shape[1]
            pad_len = max_len - curr_len
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), pad_token_id, dtype=torch.long, device=device)
                pad_msk = torch.zeros(1, pad_len, device=device)
                if padding_side == "right":
                    p_ids = torch.cat([input_ids_list[i], pad_ids], dim=1)
                    p_mask = torch.cat([attn_mask_list[i], pad_msk], dim=1)
                else:
                    p_ids = torch.cat([pad_ids, input_ids_list[i]], dim=1)
                    p_mask = torch.cat([pad_msk, attn_mask_list[i]], dim=1)
            else:
                p_ids = input_ids_list[i]
                p_mask = attn_mask_list[i]
            padded_ids.append(p_ids)
            padded_masks.append(p_mask)
        
        final_ids = torch.cat(padded_ids, dim=0)
        final_masks = torch.cat(padded_masks, dim=0)
        final_embeds = self._get_text_embeds(final_ids)
        return final_embeds, final_masks

    def _prompt_wrap_embeds(self, speech_embeds, speech_lens, prompt, multi_prompt=False, padding_side="right", prefix_embeds=None):
        """
        Wrap continuous speech embeddings with text prompts.
        """
        batch_size = speech_embeds.shape[0]
        device = speech_embeds.device
        
        input_embeds_list = []
        attn_mask_list = []
        
        if isinstance(prompts := prompt, str):
            prompts = [prompts] * batch_size

        for i, prompt in enumerate(prompts):
            p_before = ""
            p_after = ""
            
            clean_prompt = prompt.replace("<OPENSMILE>", "")
            
            if "<SpeechHere>" not in clean_prompt:
                p_before, p_after = clean_prompt, ""
            else:
                p_before, p_after = clean_prompt.split("<SpeechHere>")
            
            tokens_before = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            tokens_after = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)
            
            embeds_before = self._get_text_embeds(tokens_before.input_ids)
            embeds_after = self._get_text_embeds(tokens_after.input_ids)
            
            valid_len = speech_lens[i]
            cur_speech_embeds = speech_embeds[i:i+1, :valid_len, :]
            
            combined_embeds = torch.cat([embeds_before, cur_speech_embeds, embeds_after], dim=1)
            combined_mask = torch.cat([tokens_before.attention_mask, torch.ones(cur_speech_embeds.shape[:2], device=device), tokens_after.attention_mask], dim=1)
            
            if prefix_embeds is not None:
                curr_prefix = prefix_embeds[i:i+1]
                combined_embeds = torch.cat([curr_prefix, combined_embeds], dim=1)
                combined_mask = torch.cat([torch.ones(curr_prefix.shape[:2], device=device), combined_mask], dim=1)
            
            input_embeds_list.append(combined_embeds)
            attn_mask_list.append(combined_mask)
        
        # Padding
        max_len = max([e.shape[1] for e in input_embeds_list])
        padded_embeds = []
        padded_masks = []
        
        for i in range(batch_size):
            curr_len = input_embeds_list[i].shape[1]
            pad_len = max_len - curr_len
            if pad_len > 0:
                pad_emb = torch.zeros(1, pad_len, input_embeds_list[i].shape[2], dtype=input_embeds_list[i].dtype, device=device)
                pad_msk = torch.zeros(1, pad_len, device=device)
                if padding_side == "right":
                    p_emb = torch.cat([input_embeds_list[i], pad_emb], dim=1)
                    p_mask = torch.cat([attn_mask_list[i], pad_msk], dim=1)
                else:
                    p_emb = torch.cat([pad_emb, input_embeds_list[i]], dim=1)
                    p_mask = torch.cat([pad_msk, attn_mask_list[i]], dim=1)
            else:
                p_emb = input_embeds_list[i]
                p_mask = attn_mask_list[i]
            padded_embeds.append(p_emb)
            padded_masks.append(p_mask)
        
        final_embeds = torch.cat(padded_embeds, dim=0)
        final_masks = torch.cat(padded_masks, dim=0)
        
        return final_embeds, final_masks

    def forward(self, samples, verbose=False):
        """
        Forward pass for training.
        Targets include Chain-of-Thought analysis.
        """
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        prompts = samples.get("prompt", None)
        if prompts is None:
            raise ValueError("Prompt must be provided in samples for forward pass")
        
        # 1. Extract OpenSMILE features (Ground Truth)
        opensmile_texts = None
        if self.use_opensmile:
            if self.opensmile_extractor is None:
                from speech_encoders.opensmile_features_selected import OpenSMILESelectedExtractor
                self.opensmile_extractor = OpenSMILESelectedExtractor(selected_indices=self.opensmile_selected_indices)
            opensmile_texts = self.extract_opensmile_text(raw_wav, audio_attention_mask)

        # 2. Encode Audio
        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)
        
        # 3. Wrap Inputs (CoT mode: No OpenSMILE input injection)
        multi_prompt = isinstance(prompts, list)
        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output, speech_lens, prompts, multi_prompt=multi_prompt, padding_side="right"
        )
        
        # 4. Prepare Targets (CoT: <ANALYSE> ... </ANALYSE> Label)
        device = inputs_embeds.device
        
        full_embeds_list = []
        full_mask_list = []
        labels_list = []
        
        # Add BOS token
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        bos_tensor = torch.tensor([[bos_token_id]], device=device)
        bos_embeds = self._get_text_embeds(bos_tensor)
        
        # Prepare target strings
        # Format: <ANALYSE> {features} </ANALYSE> {label}
        target_strings = []
        labels = samples["text"]
        
        for i, label in enumerate(labels):
            cot = opensmile_texts[i] if opensmile_texts and i < len(opensmile_texts) else "No analysis available."
            # Construct target string
            # We assume the model should output the analysis, then the label.
            # Adding newlines for separation.
            full_target_str = f" {cot} </ANALYSE> {label}{self.end_sym}"
            target_strings.append(full_target_str)

        for i, tgt_text in enumerate(target_strings):
            tgt_tokens = self.tokenizer(tgt_text, return_tensors="pt", add_special_tokens=False).to(device)
            tgt_embeds = self._get_text_embeds(tgt_tokens.input_ids)
            
            curr_input_embeds = inputs_embeds[i:i+1]
            curr_mask = attention_mask[i:i+1]
            
            valid_len = curr_mask.sum(dim=1).int().item()
            real_input_embeds = curr_input_embeds[:, :valid_len]
            real_mask = curr_mask[:, :valid_len]
            
            full_emb = torch.cat([bos_embeds, real_input_embeds, tgt_embeds], dim=1)
            full_msk = torch.cat([torch.ones(1, 1, device=device), real_mask, tgt_tokens.attention_mask], dim=1)
            
            num_input_tokens = real_input_embeds.shape[1]
            label_ids = torch.cat([
                torch.full((1, 1 + num_input_tokens), -100, dtype=torch.long, device=device),
                tgt_tokens.input_ids
            ], dim=1)
            
            full_embeds_list.append(full_emb)
            full_mask_list.append(full_msk)
            labels_list.append(label_ids)
        
        # Re-pad final batch
        max_len = max([e.shape[1] for e in full_embeds_list])
        
        final_embeds_list = []
        final_masks_list = []
        final_labels_list = []
        
        for i in range(len(full_embeds_list)):
            pad_len = max_len - full_embeds_list[i].shape[1]
            if pad_len > 0:
                p_emb = torch.cat([full_embeds_list[i], torch.zeros(1, pad_len, full_embeds_list[i].shape[2], dtype=full_embeds_list[i].dtype, device=device)], dim=1)
                p_mask = torch.cat([full_mask_list[i], torch.zeros(1, pad_len, device=device)], dim=1)
                p_lbl = torch.cat([labels_list[i], torch.full((1, pad_len), -100, dtype=torch.long, device=device)], dim=1)
            else:
                p_emb = full_embeds_list[i]
                p_mask = full_mask_list[i]
                p_lbl = labels_list[i]
            
            final_embeds_list.append(p_emb)
            final_masks_list.append(p_mask)
            final_labels_list.append(p_lbl)
        
        batch_inputs_embeds = torch.cat(final_embeds_list, dim=0)
        batch_attention_mask = torch.cat(final_masks_list, dim=0)
        batch_labels = torch.cat(final_labels_list, dim=0)
        
        # Ensure dtype match
        target_dtype = torch.bfloat16
        try:
            if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "embed_tokens"):
                 target_dtype = self.llm_model.model.embed_tokens.weight.dtype
            elif hasattr(self.llm_model, "base_model") and hasattr(self.llm_model.base_model, "model"):
                 target_dtype = self.llm_model.base_model.model.embed_tokens.weight.dtype
        except:
            pass
            
        if batch_inputs_embeds.dtype != target_dtype:
            batch_inputs_embeds = batch_inputs_embeds.to(target_dtype)

        # 5. Forward Pass
        outputs = self.llm_model(
            inputs_embeds=batch_inputs_embeds,
            attention_mask=batch_attention_mask,
            return_dict=True,
            labels=batch_labels,
        )
        loss = outputs.loss
        
        return {"loss": loss}

    def get_last_token_logits(self, samples, prompts=None):
        """
        Get logits for the last token (same as eval_bf16 / opensmile_selected).
        Input: BOS + prompt + audio. One forward → logits for next token (use Real/Fake token ids for probabilities).
        """
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        if prompts is None:
            prompts = samples.get("prompt")
        if prompts is None:
            raise ValueError("Prompts must be provided")

        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)
        batch_size = speech_output.shape[0]
        device = raw_wav.device
        multi_prompt = isinstance(prompts, list)
        if not multi_prompt:
            prompts = [prompts] * batch_size

        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * bos_token_id
        bos_embeds = self._get_text_embeds(bos)

        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output, speech_lens, prompts,
            multi_prompt=True, padding_side="left", prefix_embeds=bos_embeds
        )

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        return logits

    def get_real_fake_logits(self, samples, prompts=None):
        """
        Fast path: give OpenSMILE features + " </ANALYSE> " to the model, one forward → logits for next token (Real/Fake).
        Requires use_opensmile=True and opensmile_extractor. Use for --fast in eval_bf16_cot.py.
        """
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        if prompts is None:
            prompts = samples.get("prompt")
        if prompts is None:
            raise ValueError("Prompts must be provided")
        batch_size = raw_wav.shape[0]
        device = raw_wav.device

        if not self.use_opensmile or self.opensmile_extractor is None:
            raise RuntimeError("get_real_fake_logits requires use_opensmile=True and opensmile_extractor")
        opensmile_texts = self.extract_opensmile_text(raw_wav, audio_attention_mask)

        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)
        multi_prompt = isinstance(prompts, list)
        if not multi_prompt:
            prompts = [prompts] * batch_size

        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * bos_token_id
        bos_embeds = self._get_text_embeds(bos)
        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output, speech_lens, prompts,
            multi_prompt=True, padding_side="left", prefix_embeds=bos_embeds
        )

        suffix = " </ANALYSE> "
        list_embeds = []
        list_masks = []
        for i in range(batch_size):
            valid_len = int(attention_mask[i].sum().item())
            emb_i = inputs_embeds[i : i + 1, :valid_len, :]
            msk_i = attention_mask[i : i + 1, :valid_len]
            text_suffix = (" " + opensmile_texts[i] + suffix) if opensmile_texts and i < len(opensmile_texts) else (" No analysis available." + suffix)
            tok = self.tokenizer(text_suffix, return_tensors="pt", add_special_tokens=False).to(device)
            emb_suffix = self._get_text_embeds(tok.input_ids)
            emb_i = torch.cat([emb_i, emb_suffix], dim=1)
            msk_i = torch.cat([msk_i, tok.attention_mask], dim=1)
            list_embeds.append(emb_i)
            list_masks.append(msk_i)

        max_len = max(e.shape[1] for e in list_embeds)
        padded_embeds = []
        padded_masks = []
        for i in range(batch_size):
            pad_len = max_len - list_embeds[i].shape[1]
            if pad_len > 0:
                pad_emb = torch.zeros(1, pad_len, list_embeds[i].shape[2], dtype=list_embeds[i].dtype, device=device)
                pad_msk = torch.zeros(1, pad_len, device=device)
                padded_embeds.append(torch.cat([list_embeds[i], pad_emb], dim=1))
                padded_masks.append(torch.cat([list_masks[i], pad_msk], dim=1))
            else:
                padded_embeds.append(list_embeds[i])
                padded_masks.append(list_masks[i])
        inputs_embeds = torch.cat(padded_embeds, dim=0)
        attention_mask = torch.cat(padded_masks, dim=0)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        return logits

    def generate(self, samples, generate_cfg={}, prompts=None, inject_opensmile=False):
        """
        Generate text from audio (CoT Mode).

        - inject_opensmile=False (default): Full CoT. Input = BOS + prompt + audio.
          Model generates analysis then </ANALYSE> then label. Use larger max_new_tokens.

        - inject_opensmile=True: Detection-only. Input = BOS + prompt + audio + " {OpenSMILE} </ANALYSE> ".
          Model only generates the next token(s) = "Real" or "Fake". Use small max_new_tokens (e.g. 10).
          Requires use_opensmile=True. Same conditioning as get_real_fake_logits.
        """
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)

        if prompts is None:
            if "prompt" in samples:
                prompts = samples["prompt"]
            else:
                raise ValueError("Prompts must be provided")

        # 1. Encode
        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)
        batch_size = speech_output.shape[0] if speech_output is not None else 1
        multi_prompt = isinstance(prompts, list)
        if not multi_prompt:
            prompts = [prompts] * batch_size
            multi_prompt = True

        device = raw_wav.device
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * bos_token_id
        bos_embeds = self._get_text_embeds(bos)

        # 2. Build inputs: BOS + prompt + audio
        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output,
            speech_lens,
            prompts,
            multi_prompt=multi_prompt,
            padding_side="left",
            prefix_embeds=bos_embeds
        )

        # 3. Optionally inject OpenSMILE + " </ANALYSE> " so model only generates Real/Fake
        if inject_opensmile and self.use_opensmile and self.opensmile_extractor is not None:
            opensmile_texts = self.extract_opensmile_text(raw_wav, audio_attention_mask)
            suffix = " </ANALYSE> "
            list_embeds = []
            list_masks = []
            for i in range(batch_size):
                valid_len = int(attention_mask[i].sum().item())
                emb_i = inputs_embeds[i : i + 1, :valid_len, :]
                msk_i = attention_mask[i : i + 1, :valid_len]
                text_suffix = (" " + opensmile_texts[i] + suffix) if opensmile_texts and i < len(opensmile_texts) else (" No analysis available." + suffix)
                tok = self.tokenizer(text_suffix, return_tensors="pt", add_special_tokens=False).to(device)
                emb_suffix = self._get_text_embeds(tok.input_ids)
                emb_i = torch.cat([emb_i, emb_suffix], dim=1)
                msk_i = torch.cat([msk_i, tok.attention_mask], dim=1)
                list_embeds.append(emb_i)
                list_masks.append(msk_i)
            max_len = max(e.shape[1] for e in list_embeds)
            padded_embeds = []
            padded_masks = []
            for i in range(batch_size):
                pad_len = max_len - list_embeds[i].shape[1]
                if pad_len > 0:
                    pad_emb = torch.zeros(1, pad_len, list_embeds[i].shape[2], dtype=list_embeds[i].dtype, device=device)
                    pad_msk = torch.zeros(1, pad_len, device=device)
                    padded_embeds.append(torch.cat([list_embeds[i], pad_emb], dim=1))
                    padded_masks.append(torch.cat([list_masks[i], pad_msk], dim=1))
                else:
                    padded_embeds.append(list_embeds[i])
                    padded_masks.append(list_masks[i])
            inputs_embeds = torch.cat(padded_embeds, dim=0)
            attention_mask = torch.cat(padded_masks, dim=0)
            max_new_tokens = 10
        else:
            max_new_tokens = 2600

        # 4. Generate
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=generate_cfg.get("do_sample", False),
            temperature=generate_cfg.get("temperature", 0.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
        )
        text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return text
