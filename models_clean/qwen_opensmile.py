"""
AudioLLM using Qwen2.5-Omni-3B-Instruct - FIXED VERSION

Key Fixes:
1. Proper system prompt to avoid warnings
2. Correct label masking logic
3. Config option for full tuning vs LoRA
4. Simplified and more robust input preparation
"""

import logging
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

import sys
sys.path.append("/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/models_clean")


class AudioDeepfakeLLM(nn.Module):
    """
    Qwen2.5-Omni wrapper for audio deepfake detection.
    """
    
    @property
    def device(self):
        return self.model.device

    def __init__(
        self,
        model_path="Qwen/Qwen2.5-Omni-3B-Instruct",
        
        # Training strategy
        use_lora=True,  # NEW: Set False for full fine-tuning
        freeze_audio_encoder=False,  # NEW: Set True to freeze audio tower
        
        # LoRA config (only used if use_lora=True)
        lora_rank=64,
        lora_alpha=16,
        lora_dropout=0.05,
        
        # OpenSMILE config
        use_opensmile=False, 
        
        # Resource config
        low_resource=False,
    ):
        super().__init__()
        
        self.use_opensmile = use_opensmile
        self.model_path = model_path
        self.use_lora = use_lora
        
        # ====================
        # 1. Initialize OpenSMILE
        # ====================
        self.opensmile_extractor = None
        if self.use_opensmile:
            try:
                from speech_encoders.opensmile_features import OpenSMILETextExtractor
                logging.info("Initializing OpenSMILE feature extractor...")
                self.opensmile_extractor = OpenSMILETextExtractor()
            except ImportError:
                logging.warning("Could not import OpenSMILETextExtractor. OpenSMILE features will be disabled.")

        # ====================
        # 2. Load Processor & Model
        # ====================
        logging.info(f'Loading Qwen2.5-Omni from {model_path}')
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        torch_dtype = torch.float16 if low_resource else torch.bfloat16
        
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )

        # ====================
        # 3. Optimization: Remove Vision Tower
        # ====================
        components_to_remove = ["visual", "merger", "talker", "token2wav"]
        logging.info("Applying memory optimizations (removing Vision/TTS components)...")
        for component in components_to_remove:
            if hasattr(self.model, component):
                delattr(self.model, component)
        torch.cuda.empty_cache()

        # ====================
        # 4. Freeze Audio Encoder if requested
        # ====================
        if freeze_audio_encoder:
            logging.info("Freezing audio encoder...")
            if hasattr(self.model, 'audio'):
                for param in self.model.audio.parameters():
                    param.requires_grad = False
            else:
                logging.warning("Could not find audio encoder to freeze")

        # ====================
        # 5. Setup LoRA (if enabled)
        # ====================
        if self.use_lora:
            logging.info("Setting up LoRA...")
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj",
            ]
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        else:
            logging.info("Using full fine-tuning (no LoRA)")
            # Count trainable params
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logging.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def extract_opensmile_text(self, raw_wav_list):
        """Extract OpenSMILE textual features from audio arrays."""
        if self.opensmile_extractor is None:
            return [""] * len(raw_wav_list)
        
        opensmile_texts = []
        for audio_array in raw_wav_list:
            try:
                if isinstance(audio_array, torch.Tensor):
                    audio_array = audio_array.cpu().numpy()
                
                text = self.opensmile_extractor.process_array(
                    audio_array, 
                    sample_rate=16000, 
                    precision=3
                )
                opensmile_texts.append(text)
            except Exception as e:
                logging.warning(f"OpenSMILE Extraction Failed: {e}")
                opensmile_texts.append("")
        return opensmile_texts

    def _prepare_inputs_and_labels(self, raw_wav, prompts, targets=None):
        """
        FIXED: Proper input preparation with correct label masking.
        """
        batch_size = len(raw_wav)
        
        # 1. Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts] * batch_size
            
        # 2. Extract OpenSMILE if needed
        opensmile_texts = None
        has_token = any("<OPENSMILE>" in p for p in prompts)
        
        if self.use_opensmile or has_token:
            opensmile_texts = self.extract_opensmile_text(raw_wav)

        # 3. Build conversations with PROPER SYSTEM PROMPT
        conversations = []
        
        # FIX: Use Qwen's default system prompt for audio mode
        system_prompt = (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech."
        )
                
        # KEY FIX IS HERE: Wrap system prompt in a list containing a dict
        system_content = [{"type": "text", "text": system_prompt}]

        for i, p in enumerate(prompts):
            final_prompt = p
            
            # Inject OpenSMILE if available
            if opensmile_texts and opensmile_texts[i]:
                if "<OPENSMILE>" in final_prompt:
                    final_prompt = final_prompt.replace("<OPENSMILE>", opensmile_texts[i])
                else:
                    final_prompt += f"\n\nAcoustic Analysis: {opensmile_texts[i]}"

            # Remove legacy audio tags - Qwen processor will add its own audio tokens
            # These tags are from other audio-LLM models and not needed for Qwen
            final_prompt = final_prompt.replace("<Speech>", "")
            final_prompt = final_prompt.replace("</Speech>", "")
            final_prompt = final_prompt.replace("<SpeechHere>", "")
            
            # Clean up any resulting double spaces
            final_prompt = " ".join(final_prompt.split())
            
            # Build conversation with system prompt
            # CRITICAL: For audio input, content must be a list of dicts, not a string
            msgs = [
                {"role": "system", "content": system_content},
                {
                    "role": "user", 
                    "content": [
                        {"type": "audio", "audio_url": "placeholder"},  # Signals audio input
                        {"type": "text", "text": final_prompt}
                    ]
                }
            ]
            
            # Add assistant response if training
            if targets:
                msgs.append({"role": "assistant", "content": targets[i]})
            
            conversations.append(msgs)

        # 4. Apply chat template
        texts = []
        for conv in conversations:
            text = self.processor.apply_chat_template(
                conv, 
                add_generation_prompt=(targets is None),
                tokenize=False
            )
            texts.append(text)

        # 5. Process with audio
        model_inputs = self.processor(
            text=texts,
            audios=raw_wav,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000
        ).to(self.device)

        # 6. FIX: Robust Label Masking
        labels = None
        if targets:
            input_ids = model_inputs.input_ids
            labels = input_ids.clone()
            
            # Identify the structural token sequence that separates Prompt from Response
            # For Qwen ChatML, this is "<|im_start|>assistant\n"
            sep_tokens = self.processor.tokenizer.encode(
                "<|im_start|>assistant\n", 
                add_special_tokens=False
            )
            
            # Handle potential BOS token edge case
            if len(sep_tokens) > 1 and sep_tokens[0] == self.processor.tokenizer.bos_token_id:
                 sep_tokens = sep_tokens[1:]

            batch_size = input_ids.shape[0]
            
            for i in range(batch_size):
                input_ids_i = input_ids[i].tolist()
                
                # 1. Find where the response starts
                # We search backwards to find the LAST occurrence of the separator
                # This handles multi-turn chats correctly (we only train on the last response)
                response_start_index = -1
                
                # Search loop
                n = len(sep_tokens)
                for idx in range(len(input_ids_i) - n, -1, -1):
                    if input_ids_i[idx : idx + n] == sep_tokens:
                        response_start_index = idx + n # Start of the ACTUAL response text
                        break
                
                # 2. Apply Masking
                if response_start_index != -1:
                    # MASK the Prompt (Indices 0 to start_of_response)
                    # We set these to -100 so the model ignores them during training
                    labels[i, :response_start_index] = -100
                    
                    # The Response (indices from response_start_index to end) is LEFT ALONE.
                    # It retains the original input_ids, so the model learns to predict it.
                else:
                    # Safety net: If something is wrong and we can't find the separator,
                    # mask the whole thing so we don't train on garbage.
                    labels[i, :] = -100
                    logging.warning(f"Sample {i}: Separator not found. Masking entire sample.")

            # 3. Always mask padding tokens (standard hygiene)
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return model_inputs, labels

    def forward(self, samples):
        """Forward pass for training."""
        raw_wav = samples["raw_wav"]
        prompts = samples["prompt"]
        targets = samples["text"]

        inputs, labels = self._prepare_inputs_and_labels(raw_wav, prompts, targets)

        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            input_features=inputs.get("input_features"),
            feature_attention_mask=inputs.get("feature_attention_mask"),
            labels=labels
        )
        
        return {"loss": outputs.loss}

    def get_last_token_logits(self, samples):
        """Get logits for evaluation."""
        raw_wav = samples["raw_wav"]
        prompts = samples["prompt"]

        inputs, _ = self._prepare_inputs_and_labels(raw_wav, prompts, targets=None)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                input_features=inputs.get("input_features"),
                feature_attention_mask=inputs.get("feature_attention_mask")
            )

        return outputs.logits[:, -1, :]

    def generate(self, samples, max_new_tokens=20, do_sample=False):
        """Generate text response."""
        raw_wav = samples["raw_wav"]
        prompts = samples["prompt"]

        inputs, _ = self._prepare_inputs_and_labels(raw_wav, prompts, targets=None)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                input_features=inputs.get("input_features"),
                feature_attention_mask=inputs.get("feature_attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample
            )

        # Decode only new tokens
        input_len = inputs.input_ids.shape[1]
        generated_text = self.processor.batch_decode(
            generated_ids[:, input_len:], 
            skip_special_tokens=True
        )
        
        return generated_text
