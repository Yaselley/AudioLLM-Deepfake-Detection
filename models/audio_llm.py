import logging
import contextlib
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer, 
    LlamaTokenizer, 
    Wav2Vec2Model, 
    WavLMModel, 
    WhisperFeatureExtractor,
    AutoModelForCausalLM
)

from transformers import Qwen2Tokenizer
# from transformers import Qwen3Tokenizer
from .modeling_whisper import WhisperModel
# from .modeling_qwen2 import Qwen2ForCausalLM
# from .modeling_qwen3 import Qwen3ForCausalLM
from .modeling_llama import LlamaForCausalLM

class AudioDeepfakeLLM(nn.Module):
    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        # LLM Args
        llm_path="",
        llm_type="qwen2", # qwen2, qwen3, llama
        
        # Audio Args
        audio_encoder_path="",
        audio_encoder_type="whisper", # whisper, wav2vec2, wavlm
        freeze_audio_encoder=True,
        
        # Adapter config
        freeze_adapter=False,

        # LoRA config
        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        
        # Tokenizer config
        special_tokens=None, # List of strings to add to vocab

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
        self.audio_encoder_type = audio_encoder_type.lower()
        self.llm_type = llm_type.lower()

        # ====================
        # 1. Load Tokenizer & LLM
        # ====================
        logging.info(f'Loading {self.llm_type} Tokenizer & Model from {llm_path}')
        
        if "llama" in self.llm_type:
            # self.tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.end_sym = self.tokenizer.eos_token if not end_sym else end_sym
            
            model_cls = AutoModelForCausalLM
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
        elif "qwen" in self.llm_type:
            # Determine Tokenizer Class
            TokenizerCls = AutoTokenizer
            self.tokenizer = TokenizerCls.from_pretrained(llm_path, use_fast=False, trust_remote_code=True)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.end_sym = "<|endoftext|>" if not end_sym else end_sym
            
            # Determine Model Class
            model_cls = AutoModelForCausalLM
            
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        else:
            raise ValueError(f"Unknown LLM type: {self.llm_type}")

        # Add Special Tokens
        self.added_tokens = False
        if special_tokens:
            logging.info(f"Adding special tokens: {special_tokens}")
            num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            if num_added > 0:
                self.added_tokens = True

        # Load LLM
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

        # Freeze LLM
        for param in self.llm_model.parameters():
            param.requires_grad = False
            
        # Apply LoRA
        if self.lora:
            # If we added tokens, we MUST train embeddings/head to learn them
            modules_to_save = []
            if self.added_tokens:
                logging.info("Enabling training of embeddings and lm_head for new tokens via modules_to_save")
                modules_to_save = ["embed_tokens", "lm_head"]

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
        # 2. Load Audio Encoder
        # ====================
        logging.info(f'Loading {self.audio_encoder_type} from {audio_encoder_path}')
        
        if self.audio_encoder_type == "whisper":
            self.audio_encoder = WhisperModel.from_pretrained(audio_encoder_path).encoder
            self.audio_dim = self.audio_encoder.config.d_model
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(audio_encoder_path)
            
        elif self.audio_encoder_type == "wav2vec2":
            self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_path)
            self.audio_dim = self.audio_encoder.config.hidden_size
            
        elif self.audio_encoder_type == "wavlm":
            self.audio_encoder = WavLMModel.from_pretrained(audio_encoder_path)
            self.audio_dim = self.audio_encoder.config.hidden_size
            
        else:
            raise ValueError(f"Unknown audio encoder type: {self.audio_encoder_type}")

        if freeze_audio_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            self.audio_encoder.eval()

        # Audio Projector
        # Ensure projection is in correct dtype
        self.speech_proj = nn.Linear(self.audio_dim, self.llm_model.config.hidden_size)
        if freeze_adapter:
            for param in self.speech_proj.parameters():
                param.requires_grad = False
            self.speech_proj.eval()

    def _process_audio(self, raw_wav, audio_attention_mask=None):
        """
        Extract features from raw audio based on encoder type.
        """
        device = raw_wav.device
        
        if self.audio_encoder_type == "whisper":
            # Whisper expects log-mel spectrograms of specific length
            # The WhisperModel has learned positional embeddings of fixed size (usually 1500 tokens corresponding to 30s)
            # We must pad the input to 30s so that conv output matches positional embedding size.
            
            with torch.no_grad():
                raw_wav_cpu = raw_wav.cpu().numpy()
                inputs = self.feature_extractor(
                    [w for w in raw_wav_cpu], 
                    sampling_rate=16000, 
                    return_tensors="pt",
                    padding="max_length", # Required for WhisperModel pos embeddings
                    max_length=480000, # 30s * 16000
                    truncation=True
                )
                input_features = inputs.input_features.to(device)
                
            # Forward pass through Whisper Encoder
            # This returns hidden states of shape [B, 1500, D] (for 30s audio)
            outputs = self.audio_encoder(input_features)
            hidden_states = outputs.last_hidden_state # [B, 1500, D]
            return hidden_states
            
        elif self.audio_encoder_type in ["wav2vec2", "wavlm"]:
            # Wav2Vec2/WavLM expect raw audio
            outputs = self.audio_encoder(
                raw_wav, 
                attention_mask=audio_attention_mask
            )
            return outputs.last_hidden_state
            
        return None

    def encode_speech(self, raw_wav, audio_attention_mask=None):
        with self.maybe_autocast():
            speech_embeds = self._process_audio(raw_wav, audio_attention_mask)
            speech_embeds = self.speech_proj(speech_embeds)
            return speech_embeds

    def _get_text_embeds(self, token_ids):
        """
        Helper to get embeddings depending on structure
        """
        # Try-catch to handle different model wrapping (PEFT, etc.)
        try:
            # Standard HF models usually have get_input_embeddings
            return self.llm_model.get_input_embeddings()(token_ids)
        except:
            # Fallback for nested models
            if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "embed_tokens"):
                return self.llm_model.model.embed_tokens(token_ids)
            elif hasattr(self.llm_model, "base_model") and hasattr(self.llm_model.base_model, "model"):
                return self.llm_model.base_model.model.embed_tokens(token_ids)
            try:
                return self.llm_model.model.model.embed_tokens(token_ids)
            except:
                return self.llm_model.embed_tokens(token_ids)

    def prompt_wrap(self, speech_embeds, prompt, multi_prompt=False):
        """
        Wraps speech embeddings with text prompts.
        """
        batch_size = speech_embeds.shape[0]
        device = speech_embeds.device
        
        input_embeds_list = []
        attn_mask_list = []
        
        if isinstance(prompts := prompt, str):
            prompts = [prompts] * batch_size

        for i, prompt in enumerate(prompts):
            if "<SpeechHere>" not in prompt:
                p_before, p_after = prompt, ""
            else:
                p_before, p_after = prompt.split("<SpeechHere>")
            
            tokens_before = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            tokens_after = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)
            
            embeds_before = self._get_text_embeds(tokens_before.input_ids)
            embeds_after = self._get_text_embeds(tokens_after.input_ids)
            
            cur_speech_embeds = speech_embeds[i:i+1] # [1, L, H]
            
            combined_embeds = torch.cat([embeds_before, cur_speech_embeds, embeds_after], dim=1)
            
            # Mask
            combined_mask = torch.cat([
                tokens_before.attention_mask, 
                torch.ones(cur_speech_embeds.shape[:2], device=device), 
                tokens_after.attention_mask
            ], dim=1)
            
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
                # Use same dtype as embeddings
                p_emb = torch.cat([input_embeds_list[i], torch.zeros(1, pad_len, input_embeds_list[i].shape[2], dtype=input_embeds_list[i].dtype, device=device)], dim=1)
                p_mask = torch.cat([attn_mask_list[i], torch.zeros(1, pad_len, device=device)], dim=1)
            else:
                p_emb = input_embeds_list[i]
                p_mask = attn_mask_list[i]
            
            padded_embeds.append(p_emb)
            padded_masks.append(p_mask)
            
        final_embeds = torch.cat(padded_embeds, dim=0)
        final_masks = torch.cat(padded_masks, dim=0)
        
        return final_embeds, final_masks

    def forward(self, samples, verbose=False):
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        
        # 1. Encode Audio
        speech_embeds = self.encode_speech(raw_wav, audio_attention_mask)
        
        # 2. Get Prompts
        prompts = samples.get("prompt", None)
        if prompts is None:
             raise ValueError("Prompt must be provided in samples for forward pass")

        # 3. Wrap Inputs
        multi_prompt = isinstance(prompts, list)
        inputs_embeds, attention_mask = self.prompt_wrap(speech_embeds, prompts, multi_prompt=multi_prompt)
        
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
            # Tokenize target
            tgt_tokens = self.tokenizer(tgt_text, return_tensors="pt", add_special_tokens=False).to(device)
            tgt_embeds = self._get_text_embeds(tgt_tokens.input_ids)
            
            curr_input_embeds = inputs_embeds[i] # [Max_Len, H]
            curr_mask = attention_mask[i] # [Max_Len]
            
            # Remove padding from input preparation (slice to valid length)
            valid_len = curr_mask.sum().int().item()
            real_input_embeds = curr_input_embeds[:valid_len].unsqueeze(0)
            real_mask = curr_mask[:valid_len].unsqueeze(0)
            
            # Concat: [BOS, Prompt+Speech, Target]
            full_emb = torch.cat([bos_embeds, real_input_embeds, tgt_embeds], dim=1)
            full_msk = torch.cat([torch.ones(1, 1, device=device), real_mask, tgt_tokens.attention_mask], dim=1)
            
            # Labels
            # -100 for BOS, -100 for Prompt+Speech, TokenIDs for Target
            num_input_tokens = real_input_embeds.shape[1]
            label_ids = torch.cat([
                torch.full((1, 1 + num_input_tokens), -100, dtype=torch.long, device=device), # +1 for BOS
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
        
        # 4. Forward Pass
        outputs = self.llm_model(
            inputs_embeds=batch_inputs_embeds,
            attention_mask=batch_attention_mask,
            return_dict=True,
            labels=batch_labels,
        )
        loss = outputs.loss
        
        if verbose:
            return {"loss": loss}
            
        return {"loss": loss}

    def generate(self, samples, generate_cfg={}, prompts=None):
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        
        # Encode
        speech_embeds = self.encode_speech(raw_wav, audio_attention_mask)
        
        # Wrap
        if prompts is not None:
            multi_prompt = isinstance(prompts, list)
            inputs_embeds, attention_mask = self.prompt_wrap(speech_embeds, prompts, multi_prompt=multi_prompt)
        elif "prompt" in samples:
            prompts = samples["prompt"]
            multi_prompt = isinstance(prompts, list)
            inputs_embeds, attention_mask = self.prompt_wrap(speech_embeds, prompts, multi_prompt=multi_prompt)
        else:
             raise ValueError("Prompts must be provided")
        
        # Add BOS
        batch_size = inputs_embeds.shape[0]
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        
        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.long,
            device=inputs_embeds.device,
        ) * bos_token_id
        
        bos_embeds = self._get_text_embeds(bos)
        atts_bos = torch.ones([batch_size, 1], dtype=torch.long).to(inputs_embeds.device)
        
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        # Generate
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
        return text
