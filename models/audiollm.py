"""Audio-language model for speech deepfake detection.

Architecture: a self-supervised speech encoder (XLS-R / Wav2Vec2) followed by a
single linear projector and an instruction-tuned Qwen LLM. The LLM is either
kept frozen or adapted with LoRA, and an optional MLP classification head can
be trained on the pooled final hidden state.
"""

from __future__ import annotations

import contextlib
import logging
import typing

import torch
import torch.nn as nn

if not hasattr(typing, "Self"):
    try:
        from typing_extensions import Self

        typing.Self = Self
    except ImportError:

        class _SelfType:
            def __class_getitem__(cls, item):
                return item

        typing.Self = _SelfType

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .speech_encoders import BaseSpeechEncoder, create_speech_encoder


class AudioDeepfakeLLM(nn.Module):
    """Continuous-prefix audio-language model for real/fake speech detection."""

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.amp.autocast("cuda", dtype=dtype)
        return contextlib.nullcontext()

    def __init__(
        self,
        # LLM args
        llm_path: str = "",
        llm_type: str = "qwen3.5",
        # Audio encoder args
        audio_encoder_path: str = "",
        audio_encoder_type: str = "wav2vec2",
        audio_encoder_source: str = "fairseq",
        freeze_audio_encoder: bool = True,
        # Weighted layer pooling (Wav2Vec2 only)
        use_weighted_layer_sum: bool = False,
        weighted_layer_norm: str = "none",
        # Adapter config
        freeze_adapter: bool = False,
        # LoRA config
        lora: bool = True,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        # Optional MLP adaptor on LLM hidden states
        mlp_adaptor: bool = False,
        mlp_hidden_dim: int = 768,
        mlp_dropout: float = 0.1,
        mlp_num_classes: int = 2,
        mlp_loss_weight: float = 0.5,
        # Tokenizer config
        special_tokens=None,
        # Generation config
        max_txt_len: int = 128,
        end_sym: str = "",
        low_resource: bool = False,
        device_8bit: int = 0,
    ):
        super().__init__()

        self.lora = lora
        self.max_txt_len = max_txt_len
        self.low_resource = low_resource
        self.llm_type = llm_type.lower()

        # ====================
        # 1. Load tokenizer & LLM
        # ====================
        logging.info("Loading %s tokenizer & model from %s", self.llm_type, llm_path)

        if "llama" in self.llm_type:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.end_sym = self.tokenizer.eos_token if not end_sym else end_sym
            model_cls = AutoModelForCausalLM
        elif "qwen" in self.llm_type:
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_path, use_fast=False, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.end_sym = "<|endoftext|>" if not end_sym else end_sym
            model_cls = AutoModelForCausalLM
        else:
            raise ValueError(f"Unknown LLM type: {self.llm_type}")

        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        # ====================
        # 2. Create speech encoder
        # ====================
        if not audio_encoder_type or audio_encoder_type.lower() == "none":
            raise ValueError("audio_encoder_type must be specified")

        logging.info(
            "Creating %s encoder from %s", audio_encoder_type, audio_encoder_path
        )
        encoder_kwargs = {"freeze": freeze_audio_encoder}
        if audio_encoder_type.lower() in {"wav2vec2", "wav2vec2-bert"}:
            encoder_kwargs.update(
                {
                    "use_weighted_layer_sum": use_weighted_layer_sum,
                    "weighted_layer_norm": weighted_layer_norm,
                }
            )

        self.audio_encoder: BaseSpeechEncoder = create_speech_encoder(
            encoder_type=audio_encoder_type,
            encoder_path=audio_encoder_path,
            encoder_source=audio_encoder_source,
            **encoder_kwargs,
        )

        # ====================
        # 3. Optional extra special tokens
        # ====================
        self.added_tokens = False
        if special_tokens:
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": list(special_tokens)}
            )
            if num_added > 0:
                self.added_tokens = True
                logging.info("Added %s special tokens to vocabulary", num_added)

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
                trust_remote_code=True,
            )
        else:
            self.llm_model = model_cls.from_pretrained(
                llm_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        if self.added_tokens:
            self.llm_model.resize_token_embeddings(len(self.tokenizer))

        # ====================
        # 5. Setup LoRA
        # ====================
        for param in self.llm_model.parameters():
            param.requires_grad = False

        if self.lora:
            modules_to_save = []
            if self.added_tokens:
                logging.info(
                    "Training embeddings and lm_head for the added tokens via modules_to_save"
                )
                modules_to_save = ["embed_tokens", "lm_head"]

            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
            )
            self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            self._break_tied_embedding_alias()
            self.llm_model.print_trainable_parameters()
        else:
            self._break_tied_embedding_alias()

        # ====================
        # 6. Audio projector
        # ====================
        audio_dim = self.audio_encoder.output_dim
        if audio_dim is None:
            raise ValueError(f"Encoder {audio_encoder_type} returned no output_dim")
        self.speech_proj = nn.Linear(audio_dim, self.llm_model.config.hidden_size)
        if freeze_adapter:
            for param in self.speech_proj.parameters():
                param.requires_grad = False
            self.speech_proj.eval()

        # ====================
        # 7. Optional MLP adaptor on LLM hidden states
        # ====================
        self.use_mlp_adaptor = mlp_adaptor
        self.mlp_loss_weight = float(mlp_loss_weight)
        self.mlp_adaptor = None
        if self.use_mlp_adaptor:
            self.mlp_adaptor = nn.Sequential(
                nn.Linear(self.llm_model.config.hidden_size, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_hidden_dim, mlp_num_classes),
            )
            logging.info(
                "Initialized MLP adaptor on hidden states: %s -> %s -> %s",
                self.llm_model.config.hidden_size,
                mlp_hidden_dim,
                mlp_num_classes,
            )

    def _break_tied_embedding_alias(self) -> None:
        """Avoid safetensors save failures caused by shared embedding/lm_head memory."""
        if not hasattr(self.llm_model, "config"):
            return

        if hasattr(self.llm_model.config, "tie_word_embeddings"):
            self.llm_model.config.tie_word_embeddings = False

        embed = self.llm_model.get_input_embeddings()
        output = self.llm_model.get_output_embeddings()
        if embed is None or output is None:
            return

        if embed.weight is output.weight:
            logging.warning(
                "Detected tied embedding and lm_head weights; cloning lm_head to "
                "avoid safetensors shared-memory save error."
            )
            new_lm_head = nn.Parameter(embed.weight.detach().clone())

            if hasattr(self.llm_model, "lm_head"):
                self.llm_model.lm_head.weight = new_lm_head
            elif hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "lm_head"):
                self.llm_model.model.lm_head.weight = new_lm_head
            elif hasattr(self.llm_model, "base_model") and hasattr(self.llm_model.base_model, "lm_head"):
                self.llm_model.base_model.lm_head.weight = new_lm_head
            elif (
                hasattr(self.llm_model, "base_model")
                and hasattr(self.llm_model.base_model, "model")
                and hasattr(self.llm_model.base_model.model, "lm_head")
            ):
                self.llm_model.base_model.model.lm_head.weight = new_lm_head

    def _label_to_index(self, label_text: str) -> int:
        label = str(label_text).strip().lower()
        if label in {"real", "bonafide", "human", "positive", "genuine"}:
            return 0
        if label in {"fake", "spoof", "attack", "negative", "deepfake"}:
            return 1
        if label.isdigit():
            return int(label)
        # Fallback for other string labels: convert to a stable class index.
        return abs(hash(label)) % 2

    def _mean_pool_last_hidden(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        masked = hidden_states * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return masked.sum(dim=1) / denom

    def _get_text_embeds(self, token_ids):
        """Get input embeddings regardless of the PEFT wrapping depth."""
        try:
            return self.llm_model.get_input_embeddings()(token_ids)
        except Exception:
            if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "embed_tokens"):
                return self.llm_model.model.embed_tokens(token_ids)
            if hasattr(self.llm_model, "base_model") and hasattr(self.llm_model.base_model, "model"):
                return self.llm_model.base_model.model.embed_tokens(token_ids)
            try:
                return self.llm_model.model.model.embed_tokens(token_ids)
            except Exception:
                return self.llm_model.embed_tokens(token_ids)

    def encode_speech(self, raw_wav: torch.Tensor, audio_attention_mask=None):
        """Encode raw audio and project it into the LLM embedding space.

        Returns:
            output: [B, T, d_llm] projected speech embeddings
            lengths: [B] valid encoder frame lengths
        """
        output, lengths = self.audio_encoder.encode(raw_wav, audio_attention_mask)
        with self.maybe_autocast():
            output = self.speech_proj(output)
        return output, lengths

    def prompt_wrap(
        self,
        speech_output,
        speech_lens,
        prompt,
        multi_prompt=False,
        padding_side="right",
        prefix_embeds=None,
    ):
        """Wrap continuous speech embeddings with the text prompt."""
        return self._prompt_wrap_embeds(
            speech_output,
            speech_lens,
            prompt,
            multi_prompt=multi_prompt,
            padding_side=padding_side,
            prefix_embeds=prefix_embeds,
        )

    def _prompt_wrap_embeds(
        self,
        speech_embeds,
        speech_lens,
        prompt,
        multi_prompt=False,
        padding_side="right",
        prefix_embeds=None,
    ):
        """Wrap continuous speech embeddings with text prompts.

        speech_embeds: [B, T, D] continuous embeddings
        """
        batch_size = speech_embeds.shape[0]
        device = speech_embeds.device

        input_embeds_list = []
        attn_mask_list = []

        prompts = [prompt] * batch_size if isinstance(prompt, str) else prompt

        for i, prompt_text in enumerate(prompts):
            p_before = ""
            p_after = ""

            # Apply the chat template when the tokenizer provides one.
            if (
                hasattr(self.tokenizer, "apply_chat_template")
                and self.tokenizer.chat_template is not None
            ):
                try:
                    placeholder = "___SPEECH_PLACEHOLDER___"
                    clean_prompt = (
                        prompt_text.replace("User:", "").replace("Answer:", "").strip()
                    )
                    clean_prompt = clean_prompt.replace("<SpeechHere>", placeholder)
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": clean_prompt},
                    ]
                    full_formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if placeholder in full_formatted_prompt:
                        p_before, p_after = full_formatted_prompt.split(placeholder)
                    elif "<SpeechHere>" not in prompt_text:
                        p_before, p_after = prompt_text, ""
                    else:
                        p_before, p_after = prompt_text.split("<SpeechHere>")
                except Exception:
                    if "<SpeechHere>" not in prompt_text:
                        p_before, p_after = prompt_text, ""
                    else:
                        p_before, p_after = prompt_text.split("<SpeechHere>")
            else:
                if "<SpeechHere>" not in prompt_text:
                    p_before, p_after = prompt_text, ""
                else:
                    p_before, p_after = prompt_text.split("<SpeechHere>")

            tokens_before = self.tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False
            ).to(device)
            tokens_after = self.tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False
            ).to(device)

            embeds_before = self._get_text_embeds(tokens_before.input_ids)
            embeds_after = self._get_text_embeds(tokens_after.input_ids)

            valid_len = speech_lens[i]
            cur_speech_embeds = speech_embeds[i : i + 1, :valid_len, :]

            combined_embeds = torch.cat(
                [embeds_before, cur_speech_embeds, embeds_after], dim=1
            )
            combined_mask = torch.cat(
                [
                    tokens_before.attention_mask,
                    torch.ones(cur_speech_embeds.shape[:2], device=device),
                    tokens_after.attention_mask,
                ],
                dim=1,
            )

            if prefix_embeds is not None:
                curr_prefix = prefix_embeds[i : i + 1]
                combined_embeds = torch.cat([curr_prefix, combined_embeds], dim=1)
                combined_mask = torch.cat(
                    [torch.ones(curr_prefix.shape[:2], device=device), combined_mask],
                    dim=1,
                )

            input_embeds_list.append(combined_embeds)
            attn_mask_list.append(combined_mask)

        max_len = max(e.shape[1] for e in input_embeds_list)
        padded_embeds = []
        padded_masks = []

        for i in range(batch_size):
            curr_len = input_embeds_list[i].shape[1]
            pad_len = max_len - curr_len
            if pad_len > 0:
                pad_emb = torch.zeros(
                    1,
                    pad_len,
                    input_embeds_list[i].shape[2],
                    dtype=input_embeds_list[i].dtype,
                    device=device,
                )
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

        return torch.cat(padded_embeds, dim=0), torch.cat(padded_masks, dim=0)

    def get_hidden_state_logits(self, samples, prompts=None):
        """Return pooled last-layer hidden states and MLP logits for the prompt."""
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)

        if prompts is None:
            prompts = samples.get("prompt")
        if prompts is None:
            raise ValueError("Prompts must be provided")

        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)
        multi_prompt = isinstance(prompts, list)

        batch_size = speech_output.shape[0]
        bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id
            else self.tokenizer.eos_token_id
        )
        device = raw_wav.device
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * bos_token_id
        bos_embeds = self._get_text_embeds(bos)

        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output,
            speech_lens,
            prompts,
            multi_prompt=multi_prompt,
            padding_side="left",
            prefix_embeds=bos_embeds,
        )

        with torch.no_grad() if not self.training else contextlib.nullcontext():
            llm_outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            last_hidden = (
                llm_outputs.hidden_states[-1]
                if hasattr(llm_outputs, "hidden_states")
                else llm_outputs.last_hidden_state
            )
            pooled = self._mean_pool_last_hidden(last_hidden, attention_mask)
            logits = self.mlp_adaptor(pooled.float())
            return pooled, logits

    def forward(self, samples, verbose=False):
        """Forward pass for supervised fine-tuning."""
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)

        prompts = samples.get("prompt", None)
        if prompts is None:
            raise ValueError("Prompt must be provided in samples for forward pass")

        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)

        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output,
            speech_lens,
            prompts,
            multi_prompt=isinstance(prompts, list),
            padding_side="right",
        )

        text_targets = [t + self.end_sym for t in samples["text"]]
        device = inputs_embeds.device

        full_embeds_list = []
        full_mask_list = []
        labels_list = []

        bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id
            else self.tokenizer.eos_token_id
        )
        bos_tensor = torch.tensor([[bos_token_id]], device=device)
        bos_embeds = self._get_text_embeds(bos_tensor)

        for i, tgt_text in enumerate(text_targets):
            tgt_tokens = self.tokenizer(
                tgt_text, return_tensors="pt", add_special_tokens=False
            ).to(device)
            tgt_embeds = self._get_text_embeds(tgt_tokens.input_ids)

            curr_input_embeds = inputs_embeds[i : i + 1]
            curr_mask = attention_mask[i : i + 1]

            valid_len = curr_mask.sum(dim=1).int().item()
            real_input_embeds = curr_input_embeds[:, :valid_len]
            real_mask = curr_mask[:, :valid_len]

            full_emb = torch.cat([bos_embeds, real_input_embeds, tgt_embeds], dim=1)
            full_msk = torch.cat(
                [torch.ones(1, 1, device=device), real_mask, tgt_tokens.attention_mask],
                dim=1,
            )

            num_input_tokens = real_input_embeds.shape[1]
            label_ids = torch.cat(
                [
                    torch.full(
                        (1, 1 + num_input_tokens), -100, dtype=torch.long, device=device
                    ),
                    tgt_tokens.input_ids,
                ],
                dim=1,
            )

            full_embeds_list.append(full_emb)
            full_mask_list.append(full_msk)
            labels_list.append(label_ids)

        max_len = max(e.shape[1] for e in full_embeds_list)
        final_embeds_list = []
        final_masks_list = []
        final_labels_list = []

        for i in range(len(full_embeds_list)):
            pad_len = max_len - full_embeds_list[i].shape[1]
            if pad_len > 0:
                p_emb = torch.cat(
                    [
                        full_embeds_list[i],
                        torch.zeros(
                            1,
                            pad_len,
                            full_embeds_list[i].shape[2],
                            dtype=full_embeds_list[i].dtype,
                            device=device,
                        ),
                    ],
                    dim=1,
                )
                p_mask = torch.cat(
                    [full_mask_list[i], torch.zeros(1, pad_len, device=device)], dim=1
                )
                p_lbl = torch.cat(
                    [
                        labels_list[i],
                        torch.full((1, pad_len), -100, dtype=torch.long, device=device),
                    ],
                    dim=1,
                )
            else:
                p_emb = full_embeds_list[i]
                p_mask = full_mask_list[i]
                p_lbl = labels_list[i]
            final_embeds_list.append(p_emb)
            final_masks_list.append(p_mask)
            final_labels_list.append(p_lbl)

        outputs = self.llm_model(
            inputs_embeds=torch.cat(final_embeds_list, dim=0),
            attention_mask=torch.cat(final_masks_list, dim=0),
            return_dict=True,
            labels=torch.cat(final_labels_list, dim=0),
        )
        loss = outputs.loss

        if self.use_mlp_adaptor:
            _, mlp_logits = self.get_hidden_state_logits(samples, prompts)
            cls_targets = torch.tensor(
                [self._label_to_index(t) for t in samples["text"]],
                dtype=torch.long,
                device=device,
            )
            cls_loss = torch.nn.functional.cross_entropy(mlp_logits, cls_targets)
            loss = loss + self.mlp_loss_weight * cls_loss
            return {"loss": loss, "cls_loss": cls_loss, "logits": mlp_logits}

        return {"loss": loss}

    def get_last_token_logits(self, samples, prompts=None):
        """Return next-token logits after the audio + prompt prefix."""
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)

        if prompts is None:
            if "prompt" not in samples:
                raise ValueError("Prompts must be provided")
            prompts = samples["prompt"]

        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)

        batch_size = speech_output.shape[0]
        bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id
            else self.tokenizer.eos_token_id
        )
        device = raw_wav.device
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * bos_token_id
        bos_embeds = self._get_text_embeds(bos)

        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output,
            speech_lens,
            prompts,
            multi_prompt=isinstance(prompts, list),
            padding_side="left",
            prefix_embeds=bos_embeds,
        )

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True
        )
        return outputs.logits[:, -1, :]

    def generate(self, samples, generate_cfg=None, prompts=None, return_outputs=False):
        """Generate text from audio."""
        generate_cfg = generate_cfg or {}
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)

        if prompts is None:
            if "prompt" not in samples:
                raise ValueError("Prompts must be provided")
            prompts = samples["prompt"]

        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)

        batch_size = speech_output.shape[0]
        bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id
            else self.tokenizer.eos_token_id
        )
        device = raw_wav.device
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * bos_token_id
        bos_embeds = self._get_text_embeds(bos)

        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output,
            speech_lens,
            prompts,
            multi_prompt=isinstance(prompts, list),
            padding_side="left",
            prefix_embeds=bos_embeds,
        )

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=generate_cfg.get("max_new_tokens", 20),
            do_sample=generate_cfg.get("do_sample", False),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
        )

        text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if return_outputs:
            with torch.no_grad():
                completion_ids = outputs
                if self.lora:
                    completion_embeds = self.llm_model.model.model.embed_tokens(
                        completion_ids.long()
                    )
                else:
                    completion_embeds = self.llm_model.model.embed_tokens(
                        completion_ids.long()
                    )
                full_embeds = torch.cat([inputs_embeds, completion_embeds], dim=1)
                full_atts = torch.cat([attention_mask, torch.ones_like(completion_ids)], dim=1)

            llm_outputs = self.llm_model(
                inputs_embeds=full_embeds,
                attention_mask=full_atts,
                return_dict=True,
            )
            prompt_length = inputs_embeds.size(1)
            completion_length = completion_ids.size(1)
            logits = llm_outputs.logits[
                :, prompt_length - 1 : prompt_length - 1 + completion_length, :
            ]
            return text, completion_ids, logits

        return text

    def _build_completion_inputs(self, samples, completion_ids, completion_mask=None):
        """Shared prefix+completion construction for policy/logit scoring."""
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        prompts = samples.get("prompt")
        if prompts is None:
            raise ValueError("Prompt must be provided")

        speech_output, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)
        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_output,
            speech_lens,
            prompts,
            multi_prompt=isinstance(prompts, list),
            padding_side="right",
        )

        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )
        bos = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        bos_embeds = self._get_text_embeds(bos)
        completion_embeds = self._get_text_embeds(completion_ids.long())

        full_embeds_list = []
        full_attention_list = []
        prompt_lens = []

        for i in range(batch_size):
            valid_len = attention_mask[i].sum().long().item()
            real_prompt_embeds = inputs_embeds[i : i + 1, :valid_len]
            real_prompt_mask = attention_mask[i : i + 1, :valid_len]
            curr_completion = completion_embeds[i : i + 1]

            full_embeds = torch.cat(
                [bos_embeds[i : i + 1], real_prompt_embeds, curr_completion], dim=1
            )
            if completion_mask is None:
                completion_attn = torch.ones_like(completion_ids[i : i + 1])
            else:
                completion_attn = completion_mask[i : i + 1].to(real_prompt_mask.dtype)
            full_attention = torch.cat(
                [
                    torch.ones((1, 1), dtype=real_prompt_mask.dtype, device=device),
                    real_prompt_mask,
                    completion_attn,
                ],
                dim=1,
            )
            full_embeds_list.append(full_embeds)
            full_attention_list.append(full_attention)
            prompt_lens.append(1 + valid_len)

        max_len = max(x.shape[1] for x in full_embeds_list)
        padded_embeds = []
        padded_attention = []
        for emb, attn in zip(full_embeds_list, full_attention_list):
            pad_len = max_len - emb.shape[1]
            if pad_len > 0:
                emb = torch.cat(
                    [
                        emb,
                        torch.zeros(
                            1, pad_len, emb.shape[-1], dtype=emb.dtype, device=device
                        ),
                    ],
                    dim=1,
                )
                attn = torch.cat(
                    [attn, torch.zeros(1, pad_len, dtype=attn.dtype, device=device)],
                    dim=1,
                )
            padded_embeds.append(emb)
            padded_attention.append(attn)

        return (
            torch.cat(padded_embeds, dim=0),
            torch.cat(padded_attention, dim=0),
            prompt_lens,
        )

    def compute_logits_for_completions(self, samples, completion_ids):
        """Compute per-token logits for the given completions (with gradients)."""
        full_embeds, full_attention, prompt_lens = self._build_completion_inputs(
            samples, completion_ids
        )
        outputs = self.llm_model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention,
            return_dict=True,
        )

        completion_len = completion_ids.shape[1]
        logits_list = []
        for i, prompt_len in enumerate(prompt_lens):
            logits_list.append(
                outputs.logits[
                    i : i + 1, prompt_len - 1 : prompt_len - 1 + completion_len, :
                ]
            )
        return torch.cat(logits_list, dim=0)

    def compute_logits_for_completions_with_cls(
        self, samples, completion_ids, completion_mask
    ):
        """Return completion logits and optional MLP classification logits.

        Returns:
            completion_logits: [B, completion_len, vocab_size]
            cls_logits: [B, num_classes] or None if no MLP adaptor is configured
        """
        full_embeds, full_attention, prompt_lens = self._build_completion_inputs(
            samples, completion_ids, completion_mask
        )
        outputs = self.llm_model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention,
            return_dict=True,
            output_hidden_states=self.use_mlp_adaptor,
        )

        completion_len = completion_ids.shape[1]
        logits_list = []
        for i, prompt_len in enumerate(prompt_lens):
            logits_list.append(
                outputs.logits[
                    i : i + 1, prompt_len - 1 : prompt_len - 1 + completion_len, :
                ]
            )
        completion_logits = torch.cat(logits_list, dim=0)

        cls_logits = None
        if self.use_mlp_adaptor:
            last_hidden = outputs.hidden_states[-1]
            pooled = self._mean_pool_last_hidden(last_hidden, full_attention)
            cls_logits = self.mlp_adaptor(pooled.float())
        return completion_logits, cls_logits
