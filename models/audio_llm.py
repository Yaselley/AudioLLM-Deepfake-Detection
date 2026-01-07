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

# Attempt to import specific tokenizers
try:
    from transformers import Qwen2Tokenizer
except ImportError:
    Qwen2Tokenizer = None

try:
    from transformers import Qwen3Tokenizer
except ImportError:
    Qwen3Tokenizer = None

from .modeling_whisper import WhisperModel
from .modeling_qwen2 import Qwen2ForCausalLM
try:
    from .modeling_qwen3 import Qwen3ForCausalLM
except ImportError:
    Qwen3ForCausalLM = None
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
            self.tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.end_sym = self.tokenizer.eos_token if not end_sym else end_sym
            
            model_cls = LlamaForCausalLM
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
        elif "qwen" in self.llm_type:
            # Determine Tokenizer Class
            if self.llm_type == "qwen3":
                if Qwen3Tokenizer:
                    TokenizerCls = Qwen3Tokenizer
                elif Qwen2Tokenizer:
                    logging.warning("Qwen3Tokenizer not found in transformers, falling back to Qwen2Tokenizer")
                    TokenizerCls = Qwen2Tokenizer
                else:
                    logging.warning("Qwen3Tokenizer/Qwen2Tokenizer not found, falling back to AutoTokenizer")
                    TokenizerCls = AutoTokenizer
            else: # qwen2
                if Qwen2Tokenizer:
                    TokenizerCls = Qwen2Tokenizer
                else:
                    logging.warning("Qwen2Tokenizer not found, falling back to AutoTokenizer")
                    TokenizerCls = AutoTokenizer
            
            self.tokenizer = TokenizerCls.from_pretrained(llm_path, use_fast=False, trust_remote_code=True)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.end_sym = "<|endoftext|>" if not end_sym else end_sym
            
            # Determine Model Class
            if self.llm_type == "qwen3":
                if Qwen3ForCausalLM:
                    model_cls = Qwen3ForCausalLM
                else:
                    logging.warning("Qwen3ForCausalLM not found in local models, using Qwen2ForCausalLM")
                    model_cls = Qwen2ForCausalLM
            else:
                model_cls = Qwen2ForCausalLM
            
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            raise ValueError(f"Unknown LLM type: {self.llm_type}")

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

        # Freeze LLM
        for param in self.llm_model.parameters():
            param.requires_grad = False
            
        # Apply LoRA
        if self.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=target_modules
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
            # Feature Extractor for on-the-fly processing
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
            # Whisper expects log-mel spectrograms
            # We convert raw_wav [B, T] -> [B, 80, 3000]
            with torch.no_grad():
                raw_wav_cpu = raw_wav.cpu().numpy()
                inputs = self.feature_extractor(
                    [w for w in raw_wav_cpu], 
                    sampling_rate=16000, 
                    return_tensors="pt",
                    padding="longest"
                )
                input_features = inputs.input_features.to(device)
                
            outputs = self.audio_encoder(input_features)
            return outputs.last_hidden_state
            
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

    def forward(self, samples, verbose=False):
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        
        # 1. Encode Audio
        speech_embeds = self.encode_speech(raw_wav, audio_attention_mask)
        
        # 2. Prepare Prompts & Labels
        prompts = samples.get("prompt", ["User: <SpeechHere> Is this audio real or fake? Answer: "])
        if isinstance(prompts, str):
            prompts = [prompts] * speech_embeds.shape[0]
            
        text_targets = [t + self.end_sym for t in samples["text"]]
        
        # 3. Wrap/Concat
        inputs_embeds, attention_mask, targets = self._prepare_inputs(speech_embeds, prompts, text_targets)
        
        # 4. Forward Pass
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        
        if verbose:
            return {"loss": loss}
            
        return {"loss": loss}

    def generate(self, samples, generate_cfg={}):
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        
        prompts = samples.get("prompt", ["User: <SpeechHere> Is this audio real or fake? Answer: "])
        if isinstance(prompts, str):
            prompts = [prompts] * raw_wav.shape[0]

        # Encode
        speech_embeds = self.encode_speech(raw_wav, audio_attention_mask)
        
        # Wrap
        inputs_embeds, attention_mask, _ = self._prepare_inputs(speech_embeds, prompts, targets=None)
        
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

    def _prepare_inputs(self, speech_embeds, prompts, targets=None):
        batch_size = speech_embeds.shape[0]
        device = speech_embeds.device
        
        input_embeds_list = []
        attn_mask_list = []
        labels_list = []
        
        # Helper to get embeddings depending on structure
        def get_text_embeds(token_ids):
            if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "embed_tokens"):
                 return self.llm_model.model.embed_tokens(token_ids)
            elif hasattr(self.llm_model, "base_model") and hasattr(self.llm_model.base_model, "model"):
                 return self.llm_model.base_model.model.embed_tokens(token_ids)
            try:
                return self.llm_model.model.model.embed_tokens(token_ids)
            except:
                return self.llm_model.embed_tokens(token_ids)

        for i, prompt in enumerate(prompts):
            p_before, p_after = prompt.split("<SpeechHere>")
            
            tokens_before = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            tokens_after = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)
            
            embeds_before = get_text_embeds(tokens_before.input_ids)
            embeds_after = get_text_embeds(tokens_after.input_ids)
            
            cur_speech_embeds = speech_embeds[i:i+1] # [1, T, H]
            
            combined_embeds = torch.cat([embeds_before, cur_speech_embeds, embeds_after], dim=1)
            
            combined_mask = torch.cat([
                tokens_before.attention_mask, 
                torch.ones(cur_speech_embeds.shape[:2], device=device), 
                tokens_after.attention_mask
            ], dim=1)
            
            if targets is not None:
                tgt_text = targets[i]
                tgt_tokens = self.tokenizer(tgt_text, return_tensors="pt", add_special_tokens=False).to(device)
                tgt_embeds = get_text_embeds(tgt_tokens.input_ids)
                
                full_embeds = torch.cat([combined_embeds, tgt_embeds], dim=1)
                full_mask = torch.cat([combined_mask, tgt_tokens.attention_mask], dim=1)
                
                num_prompt_tokens = combined_embeds.shape[1]
                
                label_ids = torch.cat([
                    torch.full((1, num_prompt_tokens), -100, dtype=torch.long, device=device),
                    tgt_tokens.input_ids
                ], dim=1)
                
                input_embeds_list.append(full_embeds)
                attn_mask_list.append(full_mask)
                labels_list.append(label_ids)
            else:
                input_embeds_list.append(combined_embeds)
                attn_mask_list.append(combined_mask)
        
        # Padding
        max_len = max([e.shape[1] for e in input_embeds_list])
        
        padded_embeds = []
        padded_masks = []
        padded_labels = []
        
        for i in range(batch_size):
            curr_len = input_embeds_list[i].shape[1]
            pad_len = max_len - curr_len
            
            p_emb = torch.cat([input_embeds_list[i], torch.zeros(1, pad_len, input_embeds_list[i].shape[2], device=device)], dim=1)
            p_mask = torch.cat([attn_mask_list[i], torch.zeros(1, pad_len, device=device)], dim=1)
            
            padded_embeds.append(p_emb)
            padded_masks.append(p_mask)
            
            if targets is not None:
                p_lbl = torch.cat([labels_list[i], torch.full((1, pad_len), -100, dtype=torch.long, device=device)], dim=1)
                padded_labels.append(p_lbl)
                
        final_embeds = torch.cat(padded_embeds, dim=0)
        final_masks = torch.cat(padded_masks, dim=0)
        final_labels = torch.cat(padded_labels, dim=0) if targets is not None else None
        
        return final_embeds, final_masks, final_labels
