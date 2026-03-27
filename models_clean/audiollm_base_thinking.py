"""
AudioLLM Base Model - Thinking Mode for Deepfake Detection.

Training target is the full chain-of-thought:
  <think>
  <Spectral Fidelity>...
  <Temporal Coherence>...
  <Prosodic Naturalness>...
  <Articulatory Realism>...
  <Semantic-Vocal Consistency>...
  </think>
  <answer>Real/Fake</answer>

Supports:
  - Audio-only input (encoder → LLM)
  - Audio + OpenSMILE features as text input
  - System prompt injection
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
from speech_encoders import create_speech_encoder, BaseSpeechEncoder
sys.path.append("/netscratch/yelkheir/DeepFense/DeepFense")


class AudioDeepfakeLLMThinking(nn.Module):
    """
    Audio-LLM for Deepfake Detection with Thinking (Chain-of-Thought).

    Training:
      Input: Audio + Prompt (+ optional OpenSMILE features)
      Target: <think>...\n</think>\n<answer>{label}</answer>

    Inference:
      Input: Audio + Prompt (+ optional OpenSMILE features)
      Output: Free-form generation of the full thinking + answer
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
        llm_path="",
        llm_type="qwen2",
        audio_encoder_path="",
        audio_encoder_type="whisper",
        audio_encoder_source="huggingface",
        freeze_audio_encoder=True,
        use_weighted_layer_sum: bool = False,
        weighted_layer_norm: str = "none",
        speechtokenizer_num_layers=8,
        speechtokenizer_codebook_size=1024,
        speechtokenizer_return_discrete=True,
        sqcodec_num_quantizers=4,
        sqcodec_num_levels=4096,
        dac_num_codebooks=1,
        dac_codebook_size=1024,
        dac_model_type="16khz",
        dac_return_discrete=True,
        freeze_adapter=False,
        use_opensmile=False,
        opensmile_only=False,
        lora=True,
        lora_rank=64,
        lora_alpha=16,
        lora_dropout=0.05,
        special_tokens=None,
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

        # ====================
        # 0. Initialize OpenSMILE (if needed)
        # ====================
        self.opensmile_extractor = None
        if self.use_opensmile:
            from speech_encoders.opensmile_features import OpenSMILETextExtractor
            logging.info("Initializing OpenSMILE feature extractor...")
            self.opensmile_extractor = OpenSMILETextExtractor()

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
        # 2. Create Speech Encoder
        # ====================
        if self.opensmile_only:
            logging.info("OpenSMILE-only mode: Skipping audio encoder creation")
            self.audio_encoder = None
            self.use_discrete_tokens = False
            self.speech_proj = None
        else:
            if not audio_encoder_type or audio_encoder_type.lower() == "none":
                raise ValueError("audio_encoder_type must be specified unless opensmile_only=True")

            logging.info(f'Creating {audio_encoder_type} encoder from {audio_encoder_path}')
            encoder_kwargs = {"freeze": freeze_audio_encoder}

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

            self.use_discrete_tokens = self.audio_encoder.is_discrete

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

        if not self.opensmile_only and self.use_discrete_tokens:
            if audio_encoder_type.lower() == "speechtokenizer":
                for layer_idx in range(self.speechtokenizer_num_layers):
                    for code_idx in range(1, self.speechtokenizer_codebook_size + 1):
                        all_special_tokens.append(f"<audio_{layer_idx}_{code_idx}>")
            elif audio_encoder_type.lower() == "sqcodec":
                for quantizer_idx in range(self.sqcodec_num_quantizers):
                    for level_idx in range(self.sqcodec_num_levels):
                        all_special_tokens.append(f"<sqcodec_{quantizer_idx}_{level_idx}>")
            elif audio_encoder_type.lower() == "dac":
                for codebook_idx in range(self.dac_num_codebooks):
                    for code_idx in range(self.dac_codebook_size):
                        all_special_tokens.append(f"<dac_{codebook_idx}_{code_idx}>")

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
                llm_path, torch_dtype=torch.float16, load_in_8bit=True,
                device_map={"": device_8bit}, trust_remote_code=True)
        else:
            self.llm_model = model_cls.from_pretrained(
                llm_path, torch_dtype=torch_dtype, trust_remote_code=True)

        if self.added_tokens:
            self.llm_model.resize_token_embeddings(len(self.tokenizer))

        # Build token mappings for discrete encoders
        if not self.opensmile_only and self.use_discrete_tokens:
            if audio_encoder_type.lower() == "speechtokenizer":
                self.speechtokenizer_token_map = {}
                for layer_idx in range(self.speechtokenizer_num_layers):
                    for code_idx in range(1, self.speechtokenizer_codebook_size + 1):
                        token_str = f"<audio_{layer_idx}_{code_idx}>"
                        self.speechtokenizer_token_map[(layer_idx, code_idx - 1)] = self.tokenizer.convert_tokens_to_ids(token_str)
            elif audio_encoder_type.lower() == "sqcodec":
                self.sqcodec_token_map = {}
                for quantizer_idx in range(self.sqcodec_num_quantizers):
                    for level_idx in range(self.sqcodec_num_levels):
                        token_str = f"<sqcodec_{quantizer_idx}_{level_idx}>"
                        self.sqcodec_token_map[(quantizer_idx, level_idx)] = self.tokenizer.convert_tokens_to_ids(token_str)
            elif audio_encoder_type.lower() == "dac":
                self.dac_token_map = {}
                for codebook_idx in range(self.dac_num_codebooks):
                    for code_idx in range(self.dac_codebook_size):
                        token_str = f"<dac_{codebook_idx}_{code_idx}>"
                        self.dac_token_map[(codebook_idx, code_idx)] = self.tokenizer.convert_tokens_to_ids(token_str)

        # ====================
        # 5. Setup LoRA
        # ====================
        for param in self.llm_model.parameters():
            param.requires_grad = False

        if self.lora:
            modules_to_save = []
            if self.added_tokens or (not self.opensmile_only and self.use_discrete_tokens):
                modules_to_save = ["embed_tokens", "lm_head"]

            if not self.opensmile_only and self.use_discrete_tokens and audio_encoder_type.lower() == "speechtokenizer":
                if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "embed_tokens"):
                    self.llm_model.model.embed_tokens.weight.requires_grad = True
                elif hasattr(self.llm_model, "base_model") and hasattr(self.llm_model.base_model, "model"):
                    self.llm_model.base_model.model.embed_tokens.weight.requires_grad = True
                if hasattr(self.llm_model, "lm_head"):
                    self.llm_model.lm_head.weight.requires_grad = True

            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save
            )
            self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            self.llm_model.print_trainable_parameters()

        # ====================
        # 6. Audio Projector (continuous embeddings only)
        # ====================
        if self.opensmile_only:
            self.speech_proj = None
        elif not self.use_discrete_tokens:
            audio_dim = self.audio_encoder.output_dim
            self.speech_proj = nn.Linear(audio_dim, self.llm_model.config.hidden_size)
            if freeze_adapter:
                for param in self.speech_proj.parameters():
                    param.requires_grad = False
                self.speech_proj.eval()
        else:
            self.speech_proj = None

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode_speech(self, raw_wav, audio_attention_mask=None):
        if self.opensmile_only:
            return None, None
        output, lengths = self.audio_encoder.encode(raw_wav, audio_attention_mask)
        if not self.use_discrete_tokens:
            with self.maybe_autocast():
                output = self.speech_proj(output)
        return output, lengths

    def extract_opensmile_text(self, raw_wav, audio_attention_mask=None):
        if self.opensmile_extractor is None:
            return None
        batch_size = raw_wav.shape[0]
        opensmile_texts = []
        for i in range(batch_size):
            audio_array = raw_wav[i].cpu().numpy()
            if audio_attention_mask is not None:
                valid_length = int(audio_attention_mask[i].sum().item())
                audio_array = audio_array[:valid_length]
            try:
                text_description = self.opensmile_extractor.process_array(audio_array, sample_rate=16000, precision=3)
                opensmile_texts.append(text_description)
            except Exception as e:
                logging.warning(f"Error extracting OpenSMILE features for sample {i}: {e}")
                opensmile_texts.append("")
        return opensmile_texts

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_text_embeds(self, token_ids):
        try:
            return self.llm_model.get_input_embeddings()(token_ids)
        except:
            if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "embed_tokens"):
                return self.llm_model.model.embed_tokens(token_ids)
            elif hasattr(self.llm_model, "base_model") and hasattr(self.llm_model.base_model, "model"):
                return self.llm_model.base_model.model.embed_tokens(token_ids)
            try:
                return self.llm_model.model.model.embed_tokens(token_ids)
            except:
                return self.llm_model.embed_tokens(token_ids)

    # ── Prompt Wrapping ───────────────────────────────────────────────────

    def prompt_wrap(self, speech_output, speech_lens, prompt, multi_prompt=False,
                    padding_side="right", prefix_embeds=None, opensmile_texts=None):
        if self.opensmile_only:
            return self._prompt_wrap_opensmile_only(prompt, multi_prompt, padding_side, prefix_embeds, opensmile_texts)
        elif self.use_discrete_tokens:
            return self._prompt_wrap_tokens(speech_output, speech_lens, prompt, multi_prompt, padding_side, prefix_embeds, opensmile_texts)
        else:
            return self._prompt_wrap_embeds(speech_output, speech_lens, prompt, multi_prompt, padding_side, prefix_embeds, opensmile_texts)

    def _prompt_wrap_opensmile_only(self, prompt, multi_prompt=False, padding_side="right",
                                     prefix_embeds=None, opensmile_texts=None):
        if opensmile_texts is None:
            raise ValueError("opensmile_texts must be provided for OpenSMILE-only mode")
        batch_size = len(opensmile_texts)
        device = next(self.llm_model.parameters()).device

        input_embeds_list = []
        attn_mask_list = []
        if isinstance(prompts := prompt, str):
            prompts = [prompts] * batch_size

        for i, prompt_text in enumerate(prompts):
            if opensmile_texts and opensmile_texts[i] and "<OPENSMILE>" in prompt_text:
                combined_text = prompt_text.replace("<OPENSMILE>", opensmile_texts[i])
            elif opensmile_texts and opensmile_texts[i]:
                combined_text = f"{prompt_text}\n\n{opensmile_texts[i]}"
            else:
                combined_text = prompt_text

            tokens = self.tokenizer(combined_text, return_tensors="pt", add_special_tokens=True).to(device)
            embeds = self._get_text_embeds(tokens.input_ids)
            input_embeds_list.append(embeds)
            attn_mask_list.append(tokens.attention_mask)

        max_len = max(e.shape[1] for e in input_embeds_list)
        padded_embeds, padded_masks = [], []
        for i in range(batch_size):
            pad_len = max_len - input_embeds_list[i].shape[1]
            if pad_len > 0:
                pad_emb = torch.zeros(1, pad_len, input_embeds_list[i].shape[2], dtype=input_embeds_list[i].dtype, device=device)
                pad_msk = torch.zeros(1, pad_len, device=device)
                if padding_side == "right":
                    padded_embeds.append(torch.cat([input_embeds_list[i], pad_emb], dim=1))
                    padded_masks.append(torch.cat([attn_mask_list[i], pad_msk], dim=1))
                else:
                    padded_embeds.append(torch.cat([pad_emb, input_embeds_list[i]], dim=1))
                    padded_masks.append(torch.cat([pad_msk, attn_mask_list[i]], dim=1))
            else:
                padded_embeds.append(input_embeds_list[i])
                padded_masks.append(attn_mask_list[i])

        final_embeds = torch.cat(padded_embeds, dim=0)
        final_masks = torch.cat(padded_masks, dim=0)
        if prefix_embeds is not None:
            final_embeds = torch.cat([prefix_embeds, final_embeds], dim=1)
            prefix_mask = torch.ones(prefix_embeds.shape[:2], device=device)
            final_masks = torch.cat([prefix_mask, final_masks], dim=1)
        return final_embeds, final_masks

    def _prompt_wrap_embeds(self, speech_embeds, speech_lens, prompt, multi_prompt=False,
                             padding_side="right", prefix_embeds=None, opensmile_texts=None):
        batch_size = speech_embeds.shape[0]
        device = speech_embeds.device
        input_embeds_list, attn_mask_list = [], []

        if isinstance(prompts := prompt, str):
            prompts = [prompts] * batch_size

        for i, p in enumerate(prompts):
            if opensmile_texts and opensmile_texts[i] and "<OPENSMILE>" in p:
                p = p.replace("<OPENSMILE>", opensmile_texts[i])

            if "<SpeechHere>" not in p:
                p_before, p_after = p, ""
            else:
                p_before, p_after = p.split("<SpeechHere>")

            tokens_before = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            tokens_after = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)
            embeds_before = self._get_text_embeds(tokens_before.input_ids)
            embeds_after = self._get_text_embeds(tokens_after.input_ids)

            valid_len = speech_lens[i]
            cur_speech = speech_embeds[i:i+1, :valid_len, :]
            combined = torch.cat([embeds_before, cur_speech, embeds_after], dim=1)
            mask = torch.cat([tokens_before.attention_mask, torch.ones(cur_speech.shape[:2], device=device), tokens_after.attention_mask], dim=1)

            if prefix_embeds is not None:
                curr_prefix = prefix_embeds[i:i+1]
                combined = torch.cat([curr_prefix, combined], dim=1)
                mask = torch.cat([torch.ones(curr_prefix.shape[:2], device=device), mask], dim=1)
            input_embeds_list.append(combined)
            attn_mask_list.append(mask)

        max_len = max(e.shape[1] for e in input_embeds_list)
        padded_embeds, padded_masks = [], []
        for i in range(batch_size):
            pad_len = max_len - input_embeds_list[i].shape[1]
            if pad_len > 0:
                pad_emb = torch.zeros(1, pad_len, input_embeds_list[i].shape[2], dtype=input_embeds_list[i].dtype, device=device)
                pad_msk = torch.zeros(1, pad_len, device=device)
                if padding_side == "right":
                    padded_embeds.append(torch.cat([input_embeds_list[i], pad_emb], dim=1))
                    padded_masks.append(torch.cat([attn_mask_list[i], pad_msk], dim=1))
                else:
                    padded_embeds.append(torch.cat([pad_emb, input_embeds_list[i]], dim=1))
                    padded_masks.append(torch.cat([pad_msk, attn_mask_list[i]], dim=1))
            else:
                padded_embeds.append(input_embeds_list[i])
                padded_masks.append(attn_mask_list[i])

        return torch.cat(padded_embeds, dim=0), torch.cat(padded_masks, dim=0)

    def _prompt_wrap_tokens(self, speech_codes, speech_lens, prompt, multi_prompt=False,
                             padding_side="right", prefix_embeds=None, opensmile_texts=None):
        batch_size = speech_codes.shape[0]
        device = speech_codes.device

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
            token_map = {}

        input_ids_list, attn_mask_list = [], []
        if isinstance(prompts := prompt, str):
            prompts = [prompts] * batch_size

        for i, p in enumerate(prompts):
            if opensmile_texts and opensmile_texts[i] and "<OPENSMILE>" in p:
                p = p.replace("<OPENSMILE>", opensmile_texts[i])

            if "<SpeechHere>" not in p:
                p_before, p_after = p, ""
            else:
                p_before, p_after = p.split("<SpeechHere>")

            tokens_before = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            tokens_after = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)

            valid_len = min(speech_lens[i].item(), speech_codes.shape[2])
            cur_codes = speech_codes[i, :, :valid_len]
            audio_token_ids = []
            for t in range(valid_len):
                for layer_idx in range(n_layers):
                    code_value = cur_codes[layer_idx, t].item()
                    token_id = token_map.get((layer_idx, code_value))
                    if token_id is not None:
                        audio_token_ids.append(token_id)

            audio_token_ids = torch.tensor(audio_token_ids, dtype=torch.long, device=device).unsqueeze(0)
            combined_ids = torch.cat([tokens_before.input_ids, audio_token_ids, tokens_after.input_ids], dim=1)
            combined_mask = torch.cat([tokens_before.attention_mask, torch.ones(audio_token_ids.shape, device=device), tokens_after.attention_mask], dim=1)
            input_ids_list.append(combined_ids)
            attn_mask_list.append(combined_mask)

        max_len = max(ids.shape[1] for ids in input_ids_list)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        padded_ids, padded_masks = [], []
        for i in range(batch_size):
            pad_len = max_len - input_ids_list[i].shape[1]
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), pad_token_id, dtype=torch.long, device=device)
                pad_msk = torch.zeros(1, pad_len, device=device)
                if padding_side == "right":
                    padded_ids.append(torch.cat([input_ids_list[i], pad_ids], dim=1))
                    padded_masks.append(torch.cat([attn_mask_list[i], pad_msk], dim=1))
                else:
                    padded_ids.append(torch.cat([pad_ids, input_ids_list[i]], dim=1))
                    padded_masks.append(torch.cat([pad_msk, attn_mask_list[i]], dim=1))
            else:
                padded_ids.append(input_ids_list[i])
                padded_masks.append(attn_mask_list[i])

        final_ids = torch.cat(padded_ids, dim=0)
        final_masks = torch.cat(padded_masks, dim=0)
        return self._get_text_embeds(final_ids), final_masks

    # ── Forward (Training) ────────────────────────────────────────────────

    def forward(self, samples, verbose=False):
        """
        Forward pass for training.

        samples keys:
          - raw_wav: [B, T]
          - audio_attention_mask: [B, T] (optional)
          - text: List[str] — full target string including <think>...</think><answer>...</answer>
          - prompt: str or List[str]
        """
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        prompts = samples.get("prompt", None)
        if prompts is None:
            raise ValueError("Prompt must be provided")

        prompt_list = prompts if isinstance(prompts, list) else [prompts]
        has_opensmile_token = any("<OPENSMILE>" in str(p) for p in prompt_list)

        opensmile_texts = None
        if has_opensmile_token or self.use_opensmile:
            if self.opensmile_extractor is None:
                from speech_encoders.opensmile_features import OpenSMILETextExtractor
                self.opensmile_extractor = OpenSMILETextExtractor()
            opensmile_texts = self.extract_opensmile_text(raw_wav, audio_attention_mask)

        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)

        multi_prompt = isinstance(prompts, list)
        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output, speech_lens, prompts,
            multi_prompt=multi_prompt, padding_side="right",
            opensmile_texts=opensmile_texts
        )

        # Target text already contains <think>...\n</think>\n<answer>...\n</answer>
        text_targets = [t + self.end_sym for t in samples["text"]]
        device = inputs_embeds.device

        bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        bos_tensor = torch.tensor([[bos_token_id]], device=device)
        bos_embeds = self._get_text_embeds(bos_tensor)

        full_embeds_list, full_mask_list, labels_list = [], [], []

        for i, tgt_text in enumerate(text_targets):
            tgt_tokens = self.tokenizer(tgt_text, return_tensors="pt", add_special_tokens=False).to(device)
            tgt_embeds = self._get_text_embeds(tgt_tokens.input_ids)

            curr_input = inputs_embeds[i:i+1]
            curr_mask = attention_mask[i:i+1]
            valid_len = curr_mask.sum(dim=1).int().item()
            real_input = curr_input[:, :valid_len]
            real_mask = curr_mask[:, :valid_len]

            full_emb = torch.cat([bos_embeds, real_input, tgt_embeds], dim=1)
            full_msk = torch.cat([torch.ones(1, 1, device=device), real_mask, tgt_tokens.attention_mask], dim=1)

            num_input_tokens = real_input.shape[1]
            label_ids = torch.cat([
                torch.full((1, 1 + num_input_tokens), -100, dtype=torch.long, device=device),
                tgt_tokens.input_ids
            ], dim=1)

            full_embeds_list.append(full_emb)
            full_mask_list.append(full_msk)
            labels_list.append(label_ids)

        max_len = max(e.shape[1] for e in full_embeds_list)
        final_embeds_list, final_masks_list, final_labels_list = [], [], []

        for i in range(len(full_embeds_list)):
            pad_len = max_len - full_embeds_list[i].shape[1]
            if pad_len > 0:
                p_emb = torch.cat([full_embeds_list[i], torch.zeros(1, pad_len, full_embeds_list[i].shape[2], dtype=full_embeds_list[i].dtype, device=device)], dim=1)
                p_mask = torch.cat([full_mask_list[i], torch.zeros(1, pad_len, device=device)], dim=1)
                p_lbl = torch.cat([labels_list[i], torch.full((1, pad_len), -100, dtype=torch.long, device=device)], dim=1)
            else:
                p_emb, p_mask, p_lbl = full_embeds_list[i], full_mask_list[i], labels_list[i]
            final_embeds_list.append(p_emb)
            final_masks_list.append(p_mask)
            final_labels_list.append(p_lbl)

        batch_embeds = torch.cat(final_embeds_list, dim=0)
        batch_masks = torch.cat(final_masks_list, dim=0)
        batch_labels = torch.cat(final_labels_list, dim=0)

        outputs = self.llm_model(
            inputs_embeds=batch_embeds,
            attention_mask=batch_masks,
            return_dict=True,
            labels=batch_labels,
        )
        return {"loss": outputs.loss}

    # ── Inference ─────────────────────────────────────────────────────────

    def generate(self, samples, generate_cfg={}, prompts=None):
        """
        Generate full thinking + answer from audio.
        """
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)

        if prompts is None:
            prompts = samples.get("prompt")
        if prompts is None:
            raise ValueError("Prompts must be provided")

        prompt_list = prompts if isinstance(prompts, list) else [prompts]
        has_opensmile_token = any("<OPENSMILE>" in str(p) for p in prompt_list)

        opensmile_texts = None
        if has_opensmile_token or self.use_opensmile:
            if self.opensmile_extractor is None:
                from speech_encoders.opensmile_features import OpenSMILETextExtractor
                self.opensmile_extractor = OpenSMILETextExtractor()
            opensmile_texts = self.extract_opensmile_text(raw_wav, audio_attention_mask)

        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)

        multi_prompt = isinstance(prompts, list)
        if self.opensmile_only:
            batch_size = len(opensmile_texts) if opensmile_texts else 1
        else:
            batch_size = speech_output.shape[0] if speech_output is not None else 1

        if not multi_prompt:
            prompts = [prompts] * batch_size
            multi_prompt = True

        device = raw_wav.device
        bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * bos_token_id
        bos_embeds = self._get_text_embeds(bos)

        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output, speech_lens, prompts,
            multi_prompt=multi_prompt, padding_side="left",
            prefix_embeds=bos_embeds, opensmile_texts=opensmile_texts
        )

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=generate_cfg.get("max_new_tokens", 1024),
            do_sample=generate_cfg.get("do_sample", False),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
        )
        text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return text
