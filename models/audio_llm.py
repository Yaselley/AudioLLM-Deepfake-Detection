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

    def _get_text_embeds(self, token_ids):
        """
        Helper to get embeddings depending on structure
        """
        if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "embed_tokens"):
                return self.llm_model.model.embed_tokens(token_ids)
        elif hasattr(self.llm_model, "base_model") and hasattr(self.llm_model.base_model, "model"):
                return self.llm_model.base_model.model.embed_tokens(token_ids)
        try:
            return self.llm_model.model.model.embed_tokens(token_ids)
        except:
            return self.llm_model.embed_tokens(token_ids)

    def prompt_wrap(self, speech_embeds, atts, prompt, multi_prompt=False):
        """
        Wraps speech embeddings with text prompts.
        Matches SALMONN prompt_wrap signature and logic.
        """
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    if "<SpeechHere>" not in p:
                        b, a = p, ""
                    else:
                        b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                
                p_before_tokens = self.tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False, padding="longest"
                ).to(speech_embeds.device)
                
                p_after_tokens = self.tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(speech_embeds.device)
                
                p_before_embeds = self._get_text_embeds(p_before_tokens.input_ids)
                p_after_embeds = self._get_text_embeds(p_after_tokens.input_ids)

                wrapped_embeds = torch.cat([p_before_embeds, speech_embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = speech_embeds.shape[0]
                if "<SpeechHere>" not in prompt:
                    p_before, p_after = prompt, ""
                else:
                    p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(speech_embeds.device)
                
                p_after_tokens = self.tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(speech_embeds.device)
                
                p_before_embeds = self._get_text_embeds(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self._get_text_embeds(p_after_tokens.input_ids).expand(batch_size, -1, -1)

                wrapped_embeds = torch.cat([p_before_embeds, speech_embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            return wrapped_embeds, wrapped_atts
        else:
            return speech_embeds, atts

    def forward(self, samples, verbose=False):
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        
        # 1. Encode Audio
        speech_embeds = self.encode_speech(raw_wav, audio_attention_mask)
        # Create speech attention mask (all 1s for the valid speech tokens)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
        
        # 2. Get Prompts
        prompts = samples.get("prompt", None)
        
        # 3. Wrap Inputs
        multi_prompt = isinstance(prompts, list)
        inputs_embeds, attention_mask = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=multi_prompt)
        
        # 4. Prepare Targets (Labels)
        text_targets = [t + self.end_sym for t in samples["text"]]
        
        to_regress_tokens = self.tokenizer(
            text_targets,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(inputs_embeds.device)
        
        to_regress_embeds = self._get_text_embeds(to_regress_tokens.input_ids)
        
        # Create targets with -100 for padding
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.tokenizer.pad_token_id, -100
        )
        
        # Mask out the prompt/speech part (-100)
        empty_targets = (
            torch.ones(
                [attention_mask.shape[0], attention_mask.shape[1]],
                dtype=torch.long
            ).to(inputs_embeds.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        
        batch_size = inputs_embeds.shape[0]
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id # Fallback
        
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=inputs_embeds.device,
        ) * bos_token_id
        
        bos_embeds = self._get_text_embeds(bos)
        atts_bos = torch.ones([batch_size, 1], dtype=torch.long).to(inputs_embeds.device)
        
        # Concat Final
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask, to_regress_tokens.attention_mask], dim=1)
        
        # Add BOS mask to targets (usually -100 for BOS too, or handled by shift)
        targets = torch.cat([torch.ones([batch_size, 1], dtype=torch.long).to(inputs_embeds.device).fill_(-100), targets], dim=1)

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

    def generate(self, samples, generate_cfg={}, prompts=None):
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        
        # Encode
        speech_embeds = self.encode_speech(raw_wav, audio_attention_mask)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
        
        # Wrap
        if prompts is not None:
            multi_prompt = isinstance(prompts, list)
            inputs_embeds, attention_mask = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=multi_prompt)
        elif "prompt" in samples:
            prompts = samples["prompt"]
            multi_prompt = isinstance(prompts, list)
            inputs_embeds, attention_mask = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=multi_prompt)
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
