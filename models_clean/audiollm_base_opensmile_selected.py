"""
AudioLLM Base Model (OpenSMILE Selected) - Clean refactored version using separated speech encoders.

This module provides a clean, modular implementation of AudioDeepfakeLLM
that uses the speech_encoders package for all audio encoding logic.
It specifically uses OpenSMILESelectedExtractor for feature extraction.
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
    Audio-LLM for Deepfake Detection (OpenSMILE Selected Features).
    
    Uses separated speech encoders for clean, modular design.
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
        use_opensmile=False,  # Whether to enable OpenSMILE (can also be auto-activated by <OPENSMILE> token in prompt)
        opensmile_only=False,  # If True, only use OpenSMILE (no encoder)
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
        self.use_opensmile = use_opensmile or opensmile_only
        self.opensmile_only = opensmile_only
        
        # Set default selected indices if not provided
        if opensmile_selected_indices is None:
            self.opensmile_selected_indices = [1, 18, 19, 20, 21, 39, 45, 51, 58, 60, 67, 77, 78, 82, 85, 86]
        else:
            self.opensmile_selected_indices = opensmile_selected_indices

        # ====================
        # 0. Initialize OpenSMILE (if needed)
        # ====================
        self.opensmile_extractor = None
        if self.use_opensmile:
            from speech_encoders.opensmile_features_selected import OpenSMILESelectedExtractor
            logging.info(f"Initializing OpenSMILE selected feature extractor with indices: {self.opensmile_selected_indices}")
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
        # If opensmile_only, skip encoder creation
        if self.opensmile_only:
            logging.info("OpenSMILE-only mode: Skipping audio encoder creation")
            self.audio_encoder = None
            self.use_discrete_tokens = False
            self.speech_proj = None
        else:
            if not audio_encoder_type or audio_encoder_type.lower() == "none":
                raise ValueError("audio_encoder_type must be specified unless opensmile_only=True")
            
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
        # 3. Add Special Tokens (for discrete encoders)
        # ====================
        self.added_tokens = False
        all_special_tokens = []
        
        if not self.opensmile_only and self.use_discrete_tokens and audio_encoder_type.lower() == "speechtokenizer":
            # Add SpeechTokenizer tokens: <audio_{layer}_{code}>
            for layer_idx in range(self.speechtokenizer_num_layers):
                for code_idx in range(1, self.speechtokenizer_codebook_size + 1):
                    all_special_tokens.append(f"<audio_{layer_idx}_{code_idx}>")
            logging.info(f"Adding {len(all_special_tokens)} SpeechTokenizer tokens")
            
        elif self.use_discrete_tokens and audio_encoder_type.lower() == "sqcodec":
            # Add SQ-Codec tokens: <sqcodec_{quantizer}_{level}>
            for quantizer_idx in range(self.sqcodec_num_quantizers):
                for level_idx in range(self.sqcodec_num_levels):
                    all_special_tokens.append(f"<sqcodec_{quantizer_idx}_{level_idx}>")
            logging.info(f"Adding {len(all_special_tokens)} SQ-Codec tokens")
                
        elif self.use_discrete_tokens and audio_encoder_type.lower() == "dac":
            for codebook_idx in range(self.dac_num_codebooks):
                for code_idx in range(self.dac_codebook_size):
                    all_special_tokens.append(f"<dac_{codebook_idx}_{code_idx}>")
            logging.info(f"Adding {len(all_special_tokens)} DAC tokens")

                    
        if special_tokens:
            all_special_tokens.extend(special_tokens)
        
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
        if not self.opensmile_only and self.use_discrete_tokens and audio_encoder_type.lower() == "speechtokenizer":
            self.speechtokenizer_token_map = {}
            for layer_idx in range(self.speechtokenizer_num_layers):
                for code_idx in range(1, self.speechtokenizer_codebook_size + 1):
                    token_str = f"<audio_{layer_idx}_{code_idx}>"
                    token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                    self.speechtokenizer_token_map[(layer_idx, code_idx - 1)] = token_id
            logging.info(f"Built token mapping for {len(self.speechtokenizer_token_map)} (layer, code) pairs")
            
        elif self.use_discrete_tokens and audio_encoder_type.lower() == "sqcodec":
            self.sqcodec_token_map = {}
            for quantizer_idx in range(self.sqcodec_num_quantizers):
                for level_idx in range(self.sqcodec_num_levels):
                    token_str = f"<sqcodec_{quantizer_idx}_{level_idx}>"
                    token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                    self.sqcodec_token_map[(quantizer_idx, level_idx)] = token_id
            logging.info(f"Built SQ-Codec token mapping for {len(self.sqcodec_token_map)} (quantizer, level) pairs")
        
        elif self.use_discrete_tokens and audio_encoder_type.lower() == "dac":
            self.dac_token_map = {}
            for codebook_idx in range(self.dac_num_codebooks):
                for code_idx in range(self.dac_codebook_size):
                    token_str = f"<dac_{codebook_idx}_{code_idx}>"
                    token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                    self.dac_token_map[(codebook_idx, code_idx)] = token_id
            logging.info(f"Built DAC token mapping for {len(self.dac_token_map)} (codebook, code) pairs")

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
            if not self.opensmile_only and self.use_discrete_tokens and audio_encoder_type.lower() == "speechtokenizer":
                logging.info("SpeechTokenizer discrete mode: Enabling full training of embedding matrix and LM head")
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
        if self.opensmile_only:
            # No projector needed for OpenSMILE-only mode
            self.speech_proj = None
        elif not self.use_discrete_tokens:
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
        
        Args:
            raw_wav: [B, T] raw audio
            audio_attention_mask: [B, T] attention mask
        
        Returns:
            output: Either [B, T, D] embeddings or [B, n_layers, T] discrete codes, or None if opensmile_only
            lengths: [B] valid lengths, or None if opensmile_only
        """
        # If OpenSMILE-only mode, return None (no encoder output)
        if self.opensmile_only:
            return None, None
        
        # Use the separated encoder - much cleaner!
        output, lengths = self.audio_encoder.encode(raw_wav, audio_attention_mask)
        
        # Project continuous embeddings to LLM space
        if not self.use_discrete_tokens:
            with self.maybe_autocast():
                output = self.speech_proj(output)
        
        return output, lengths
    
    def extract_opensmile_text(self, raw_wav: torch.Tensor, audio_attention_mask=None):
        """
        Extract OpenSMILE textual features from audio.
        
        Args:
            raw_wav: [B, T] raw audio
            audio_attention_mask: [B, T] attention mask (optional)
        
        Returns:
            List of text descriptions (one per batch item)
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
                opensmile_texts.append("")  # Empty string on error
        
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

    def prompt_wrap(self, speech_output, speech_lens, prompt, multi_prompt=False, padding_side="right", prefix_embeds=None, opensmile_texts=None):
        """
        Wrap speech embeddings/tokens with text prompts.
        
        Args:
            speech_output: Either [B, T, D] embeddings or [B, n_layers, T] discrete codes, or None if opensmile_only
            speech_lens: [B] valid lengths, or None if opensmile_only
            prompt: String or list of strings
            multi_prompt: Whether prompt is a list
            padding_side: "left" or "right"
            prefix_embeds: Optional prefix embeddings [B, L, D]
            opensmile_texts: Optional list of OpenSMILE text descriptions (one per batch item)
        
        Returns:
            input_embeds: [B, L, D] combined embeddings
            attention_mask: [B, L] attention mask
        """
        if self.opensmile_only:
            return self._prompt_wrap_opensmile_only(prompt, multi_prompt, padding_side, prefix_embeds, opensmile_texts)
        elif self.use_discrete_tokens:
            return self._prompt_wrap_tokens(speech_output, speech_lens, prompt, multi_prompt, padding_side, prefix_embeds, opensmile_texts)
        else:
            return self._prompt_wrap_embeds(speech_output, speech_lens, prompt, multi_prompt, padding_side, prefix_embeds, opensmile_texts)

    def _prompt_wrap_opensmile_only(self, prompt, multi_prompt=False, padding_side="right", prefix_embeds=None, opensmile_texts=None):
        """
        Wrap OpenSMILE text descriptions with prompts (no encoder).
        
        Args:
            prompt: String or list of strings
            multi_prompt: Whether prompt is a list
            padding_side: "left" or "right"
            prefix_embeds: Optional prefix embeddings [B, L, D]
            opensmile_texts: List of OpenSMILE text descriptions (one per batch item)
        
        Returns:
            input_embeds: [B, L, D] combined embeddings
            attention_mask: [B, L] attention mask
        """
        if opensmile_texts is None:
            raise ValueError("opensmile_texts must be provided for OpenSMILE-only mode")
        
        batch_size = len(opensmile_texts)
        device = next(self.llm_model.parameters()).device
        
        input_embeds_list = []
        attn_mask_list = []
        
        if isinstance(prompts := prompt, str):
            prompts = [prompts] * batch_size
        
        for i, prompt_text in enumerate(prompts):
            # Replace <OPENSMILE> token with actual OpenSMILE text
            if opensmile_texts and opensmile_texts[i] and "<OPENSMILE>" in prompt_text:
                combined_text = prompt_text.replace("<OPENSMILE>", opensmile_texts[i])
            elif opensmile_texts and opensmile_texts[i]:
                # Fallback: append if no token but OpenSMILE text available
                combined_text = f"{prompt_text}\n\n{opensmile_texts[i]}"
            else:
                combined_text = prompt_text
            
            tokens = self.tokenizer(combined_text, return_tensors="pt", add_special_tokens=True).to(device)
            embeds = self._get_text_embeds(tokens.input_ids)
            
            input_embeds_list.append(embeds)
            attn_mask_list.append(tokens.attention_mask)
        
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
        
        if prefix_embeds is not None:
            final_embeds = torch.cat([prefix_embeds, final_embeds], dim=1)
            prefix_mask = torch.ones(prefix_embeds.shape[:2], device=device)
            final_masks = torch.cat([prefix_mask, final_masks], dim=1)
        
        return final_embeds, final_masks

    def _prompt_wrap_tokens(self, speech_codes, speech_lens, prompt, multi_prompt=False, padding_side="right", prefix_embeds=None, opensmile_texts=None):
        """
        Wrap discrete codes with text prompts.
        speech_codes: [B, n_layers, T] discrete code values
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
            
            # Apply chat template if available
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                try:
                    placeholder = "___SPEECH_PLACEHOLDER___"
                    clean_prompt = prompt.replace("User:", "").replace("Answer:", "").strip()
                    clean_prompt = clean_prompt.replace("<SpeechHere>", placeholder)
                    
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": clean_prompt}
                    ]
                    
                    full_formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    if placeholder in full_formatted_prompt:
                        p_before, p_after = full_formatted_prompt.split(placeholder)
                    else:
                        if "<SpeechHere>" not in prompt:
                            p_before, p_after = prompt, ""
                        else:
                            p_before, p_after = prompt.split("<SpeechHere>")
                except Exception as e:
                    if "<SpeechHere>" not in prompt:
                        p_before, p_after = prompt, ""
                    else:
                        p_before, p_after = prompt.split("<SpeechHere>")
            else:
                if "<SpeechHere>" not in prompt:
                    p_before, p_after = prompt, ""
                else:
                    p_before, p_after = prompt.split("<SpeechHere>")
            
            # Replace <OPENSMILE> token with actual OpenSMILE text if available
            if opensmile_texts and opensmile_texts[i] and "<OPENSMILE>" in prompt:
                # Replace in both parts if needed
                p_before = p_before.replace("<OPENSMILE>", opensmile_texts[i])
                p_after = p_after.replace("<OPENSMILE>", opensmile_texts[i])
            # Tokenize text parts
            tokens_before = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            tokens_after = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)
            
            # Convert discrete codes to token IDs
            valid_len = speech_lens[i].item()
            actual_code_len = speech_codes.shape[2]
            valid_len = min(valid_len, actual_code_len)
            
            cur_codes = speech_codes[i, :, :valid_len]  # [n_layers, valid_len]
            
            # Convert codes to token IDs (interleave layers)
            audio_token_ids = []
            for t in range(valid_len):
                for layer_idx in range(n_layers):
                    code_value = cur_codes[layer_idx, t].item()
                    
                    # Get token ID from mapping
                    if hasattr(self, 'speechtokenizer_codebook_size'):
                        if code_value < 0 or code_value >= self.speechtokenizer_codebook_size:
                            logging.warning(f"Invalid code value {code_value}, skipping")
                            continue
                        token_id = token_map.get((layer_idx, code_value))
                    elif hasattr(self, 'sqcodec_num_levels'):
                        if code_value < 0 or code_value >= self.sqcodec_num_levels:
                            logging.warning(f"Invalid code value {code_value}, skipping")
                            continue
                        token_id = token_map.get((layer_idx, code_value))
                    elif hasattr(self, 'dac_num_codebooks'):  # ADD THIS
                        if code_value < 0 or code_value >= self.dac_codebook_size:
                            logging.warning(f"Invalid code value {code_value}, skipping")
                            continue
                        token_id = token_map.get((layer_idx, code_value))
                    else:
                        token_id = token_map.get((layer_idx, code_value))
                    
                    if token_id is None:
                        # Fallback: construct token string
                        if hasattr(self, 'speechtokenizer_codebook_size'):
                            token_str = f"<audio_{layer_idx}_{code_value + 1}>"
                        elif hasattr(self, 'dac_num_codebooks'):
                            token_str = f"<dac_{layer_idx}_{code_value}>"
                        else:
                            token_str = f"<sqcodec_{layer_idx}_{code_value}>"
                        token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                        if token_id == self.tokenizer.unk_token_id:
                            logging.warning(f"Token '{token_str}' not found, skipping")
                            continue
                    
                    audio_token_ids.append(token_id)
            
            audio_token_ids = torch.tensor(audio_token_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            # Combine: [before_tokens, audio_tokens, after_tokens]
            combined_ids = torch.cat([
                tokens_before.input_ids,
                audio_token_ids,
                tokens_after.input_ids
            ], dim=1)
            
            combined_mask = torch.cat([
                tokens_before.attention_mask,
                torch.ones(audio_token_ids.shape, device=device),
                tokens_after.attention_mask
            ], dim=1)
            
            input_ids_list.append(combined_ids)
            attn_mask_list.append(combined_mask)
        
        # Padding
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
        
        # Convert token IDs to embeddings
        final_embeds = self._get_text_embeds(final_ids)
        
        return final_embeds, final_masks

    def _prompt_wrap_embeds(self, speech_embeds, speech_lens, prompt, multi_prompt=False, padding_side="right", prefix_embeds=None, opensmile_texts=None):
        """
        Wrap continuous speech embeddings with text prompts.
        speech_embeds: [B, T, D] continuous embeddings
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
            
            # Apply chat template if available
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                try:
                    placeholder = "___SPEECH_PLACEHOLDER___"
                    clean_prompt = prompt.replace("User:", "").replace("Answer:", "").strip()
                    clean_prompt = clean_prompt.replace("<SpeechHere>", placeholder)
                    
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": clean_prompt}
                    ]
                    
                    full_formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    if placeholder in full_formatted_prompt:
                        p_before, p_after = full_formatted_prompt.split(placeholder)
                    else:
                        if "<SpeechHere>" not in prompt:
                            p_before, p_after = prompt, ""
                        else:
                            p_before, p_after = prompt.split("<SpeechHere>")
                except Exception as e:
                    if "<SpeechHere>" not in prompt:
                        p_before, p_after = prompt, ""
                    else:
                        p_before, p_after = prompt.split("<SpeechHere>")
            else:
                if "<SpeechHere>" not in prompt:
                    p_before, p_after = prompt, ""
                else:
                    p_before, p_after = prompt.split("<SpeechHere>")
            
            # Replace <OPENSMILE> token with actual OpenSMILE text if available
            if opensmile_texts and opensmile_texts[i] and "<OPENSMILE>" in prompt:
                # Replace in both parts if needed
                p_before = p_before.replace("<OPENSMILE>", opensmile_texts[i])
                p_after = p_after.replace("<OPENSMILE>", opensmile_texts[i])

            tokens_before = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            tokens_after = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)
            
            embeds_before = self._get_text_embeds(tokens_before.input_ids)
            embeds_after = self._get_text_embeds(tokens_after.input_ids)
            
            # Slice speech embeds to valid length
            valid_len = speech_lens[i]
            cur_speech_embeds = speech_embeds[i:i+1, :valid_len, :]
            
            combined_embeds = torch.cat([embeds_before, cur_speech_embeds, embeds_after], dim=1)
            
            combined_mask = torch.cat([
                tokens_before.attention_mask,
                torch.ones(cur_speech_embeds.shape[:2], device=device),
                tokens_after.attention_mask
            ], dim=1)
            
            # Add prefix if provided
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
        
        Args:
            samples: Dict with keys:
                - raw_wav: [B, T] audio
                - audio_attention_mask: [B, T] mask (optional)
                - text: List of target text strings
                - prompt: String or list of prompt strings
        
        Returns:
            Dict with 'loss' key
        """
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        
        # 1. Get Prompts and check for <OPENSMILE> token
        prompts = samples.get("prompt", None)
        if prompts is None:
            raise ValueError("Prompt must be provided in samples for forward pass")
        
        # Check if <OPENSMILE> token is present in prompt(s)
        prompt_list = prompts if isinstance(prompts, list) else [prompts]
        has_opensmile_token = any("<OPENSMILE>" in str(p) for p in prompt_list)
        
        # 2. Extract OpenSMILE features if <OPENSMILE> token is present or use_opensmile is enabled
        opensmile_texts = None
        if has_opensmile_token or self.use_opensmile:
            if self.opensmile_extractor is None:
                logging.warning("<OPENSMILE> token found in prompt but OpenSMILE extractor not initialized. Initializing now...")
                from speech_encoders.opensmile_features_selected import OpenSMILESelectedExtractor
                self.opensmile_extractor = OpenSMILESelectedExtractor(selected_indices=self.opensmile_selected_indices)
            opensmile_texts = self.extract_opensmile_text(raw_wav, audio_attention_mask)

        # 3. Encode Audio (using separated encoder, unless opensmile_only)
        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)
        
        # 4. Wrap Inputs
        multi_prompt = isinstance(prompts, list)
        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output, speech_lens, prompts, multi_prompt=multi_prompt, padding_side="right", opensmile_texts=opensmile_texts
        )
        
        # 4. Prepare Targets (Labels)
        text_targets = [t + self.end_sym for t in samples["text"]]
        device = inputs_embeds.device
        
        full_embeds_list = []
        full_mask_list = []
        labels_list = []
        
        # Add BOS token
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        bos_tensor = torch.tensor([[bos_token_id]], device=device)
        bos_embeds = self._get_text_embeds(bos_tensor)
        
        for i, tgt_text in enumerate(text_targets):
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
                p_emb = torch.cat([
                    full_embeds_list[i],
                    torch.zeros(1, pad_len, full_embeds_list[i].shape[2], dtype=full_embeds_list[i].dtype, device=device)
                ], dim=1)
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
        
        # Ensure inputs_embeds matches the model's expected dtype (usually bfloat16 for this setup)
        # We check the dtype of the first linear layer in the model to be sure
        target_dtype = torch.bfloat16 # Default to bfloat16 as per init
        
        # Try to infer from model
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
        Get logits for the last token (for inference/evaluation).
        
        Args:
            samples: Dict with raw_wav and optional audio_attention_mask
            prompts: Optional prompts (if not in samples)
        
        Returns:
            logits: [B, vocab_size] logits for last token
        """
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        
        # Prepare Prompts
        if prompts is None:
            if "prompt" in samples:
                prompts = samples["prompt"]
            else:
                raise ValueError("Prompts must be provided")
        
        # Check if <OPENSMILE> token is present in prompt(s)
        prompt_list = prompts if isinstance(prompts, list) else [prompts]
        has_opensmile_token = any("<OPENSMILE>" in str(p) for p in prompt_list)
        
        # Extract OpenSMILE features if <OPENSMILE> token is present or use_opensmile is enabled
        opensmile_texts = None
        if has_opensmile_token or self.use_opensmile:
            if self.opensmile_extractor is None:
                logging.warning("<OPENSMILE> token found in prompt but OpenSMILE extractor not initialized. Initializing now...")
                from speech_encoders.opensmile_features_selected import OpenSMILESelectedExtractor
                self.opensmile_extractor = OpenSMILESelectedExtractor(selected_indices=self.opensmile_selected_indices)
            opensmile_texts = self.extract_opensmile_text(raw_wav, audio_attention_mask)
        
        # Encode
        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)
        
        multi_prompt = isinstance(prompts, list)
        
        # Prepare BOS
        if self.opensmile_only:
            batch_size = len(opensmile_texts) if opensmile_texts else 1
        else:
            batch_size = speech_output.shape[0] if speech_output is not None else len(opensmile_texts) if opensmile_texts else 1
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        
        device = raw_wav.device
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * bos_token_id
        bos_embeds = self._get_text_embeds(bos)
        
        # Wrap with Left Padding and Prefix (BOS)
        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output,
            speech_lens,
            prompts,
            multi_prompt=multi_prompt,
            padding_side="left",
            prefix_embeds=bos_embeds,
            opensmile_texts=opensmile_texts
        )
        
        # Forward pass
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get logits of the last token
        last_token_logits = outputs.logits[:, -1, :]
        
        return last_token_logits

    def generate(self, samples, generate_cfg={}, prompts=None):
        """
        Generate text from audio.
        
        Args:
            samples: Dict with raw_wav and optional audio_attention_mask
            generate_cfg: Generation config dict
            prompts: Optional prompts
        
        Returns:
            List of generated text strings
        """
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        
        # Prepare Prompts
        if prompts is None:
            if "prompt" in samples:
                prompts = samples["prompt"]
            else:
                raise ValueError("Prompts must be provided")
        
        # Check if <OPENSMILE> token is present in prompt(s)
        prompt_list = prompts if isinstance(prompts, list) else [prompts]
        has_opensmile_token = any("<OPENSMILE>" in str(p) for p in prompt_list)
        
        # Extract OpenSMILE features if <OPENSMILE> token is present or use_opensmile is enabled
        opensmile_texts = None
        if has_opensmile_token or self.use_opensmile:
            if self.opensmile_extractor is None:
                logging.warning("<OPENSMILE> token found in prompt but OpenSMILE extractor not initialized. Initializing now...")
                from speech_encoders.opensmile_features_selected import OpenSMILESelectedExtractor
                self.opensmile_extractor = OpenSMILESelectedExtractor(selected_indices=self.opensmile_selected_indices)
            opensmile_texts = self.extract_opensmile_text(raw_wav, audio_attention_mask)
        
        # Encode
        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)
        
        multi_prompt = isinstance(prompts, list)
        
        # Prepare BOS
        if self.opensmile_only:
            batch_size = len(opensmile_texts) if opensmile_texts else 1
        else:
            batch_size = speech_output.shape[0] if speech_output is not None else len(opensmile_texts) if opensmile_texts else 1
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        
        device = raw_wav.device
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * bos_token_id
        bos_embeds = self._get_text_embeds(bos)
        
        # Wrap with Left Padding and Prefix (BOS)
        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output,
            speech_lens,
            prompts,
            multi_prompt=multi_prompt,
            padding_side="left",
            prefix_embeds=bos_embeds,
            opensmile_texts=opensmile_texts
        )
        
        # Generate
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=generate_cfg.get("max_new_tokens", 20),
            do_sample=generate_cfg.get("do_sample", False),
            temperature=generate_cfg.get("temperature", 0.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
        )
        
        text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return text
