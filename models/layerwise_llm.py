import logging
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Python 3.10 compatibility: Self is only available in 3.11+
# Patch typing.Self before importing libraries that might use it (e.g., sq_codec)
import typing
if not hasattr(typing, 'Self'):
    try:
        from typing_extensions import Self
        typing.Self = Self
    except ImportError:
        # Fallback: create a dummy Self type for runtime (not perfect but prevents ImportError)
        class _SelfType:
            def __class_getitem__(cls, item):
                return item
        typing.Self = _SelfType

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer, 
    LlamaTokenizer, 
    Wav2Vec2Model, 
    WavLMModel,
    Wav2Vec2BertModel,
    WhisperFeatureExtractor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoFeatureExtractor
)

from transformers import Qwen2Tokenizer
from .modeling_whisper import WhisperModel
from .modeling_llama import LlamaForCausalLM

# Add DeepFense path for custom modules if needed
sys.path.append("/netscratch/yelkheir/DeepFense/DeepFense")

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
        audio_encoder_type="speechtokenizer", # speechtokenizer, sqcodec, whisper, wav2vec2, wavlm
        audio_encoder_source="huggingface", # huggingface, fairseq, unil
        freeze_audio_encoder=True,
        speechtokenizer_num_layers=8, # Number of RVQ layers in SpeechTokenizer
        speechtokenizer_codebook_size=1024, # Codebook size for SpeechTokenizer (default 1024)
        sqcodec_num_quantizers=4, # Number of quantizers for SQ-Codec (e.g., 4 for SQ-SMA-16)
        sqcodec_num_levels=4096, # Number of quantization levels per quantizer for SQ-Codec (default 4096)
        
        # Adapter config
        freeze_adapter=False,

        # Intermediate layers config (for wav2vec2 and whisper)
        use_intermediate_layers=False,  # Whether to use intermediate layers instead of just last layer
        layer_weights=None,  # List of weights for each layer (will be normalized to sum to 2). If None, use uniform weights
        selected_layers=None,  # List of layer indices to use (e.g., [0, 4, 8, 12]). If None, use all layers
        
        # LoRA config
        lora=True,
        lora_rank=64,
        lora_alpha=16,
        lora_dropout=0.05,
        
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
        self.audio_encoder_source = audio_encoder_source.lower()
        self.llm_type = llm_type.lower()
        self.use_speechtokenizer = (self.audio_encoder_type == "speechtokenizer")
        self.use_sqcodec = (self.audio_encoder_type == "sqcodec")
        self.use_discrete_tokens = self.use_speechtokenizer or self.use_sqcodec
        self.speechtokenizer_num_layers = speechtokenizer_num_layers if self.use_speechtokenizer else 0
        self.speechtokenizer_codebook_size = speechtokenizer_codebook_size if self.use_speechtokenizer else None
        self.sqcodec_num_quantizers = sqcodec_num_quantizers if self.use_sqcodec else 0
        self.sqcodec_num_levels = sqcodec_num_levels if self.use_sqcodec else None
        
        # Intermediate layers config
        self.use_intermediate_layers = use_intermediate_layers
        self.selected_layers = selected_layers
        self.layer_weights = None  # Will be initialized after loading audio encoder
        
        # Only enable for wav2vec2 and whisper
        if self.use_intermediate_layers and self.audio_encoder_type not in ["wav2vec2", "whisper", "whisper_ner"]:
            logging.warning(f"Intermediate layers only supported for wav2vec2 and whisper, not {self.audio_encoder_type}. Disabling.")
            self.use_intermediate_layers = False

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
        speechtokenizer_tokens = []
        sqcodec_tokens = []
        self.speechtokenizer_codebook_size = None
        self.sqcodec_num_quantizers = None
        self.sqcodec_num_levels = None
        self.speechtokenizer_token_map = {}  # Mapping from (layer, code) -> token_id
        self.sqcodec_token_map = {}  # Mapping from (quantizer, level) -> token_id
        
        # Add SpeechTokenizer tokens if using SpeechTokenizer
        if self.use_speechtokenizer:
            # Load config early to get codebook size and num layers
            if audio_encoder_path:
                import json
                # audio_encoder_path can be a directory containing config.json and .pt file
                if os.path.isdir(audio_encoder_path):
                    config_path = os.path.join(audio_encoder_path, "config.json")
                else:
                    # Assume it's a config file path, find .pt in same directory
                    config_path = audio_encoder_path
                    audio_encoder_path = os.path.dirname(audio_encoder_path)
                
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            st_config = json.load(f)
                        logging.info(f"Loaded SpeechTokenizer config: {self.speechtokenizer_num_layers} layers, codebook_size {self.speechtokenizer_codebook_size}")
                    except Exception as e:
                        logging.warning(f"Could not load SpeechTokenizer config from {config_path}: {e}, using defaults")
            
            # Use defaults if not set
            if self.speechtokenizer_codebook_size is None:
                self.speechtokenizer_codebook_size = 1024  # Default fallback
            if self.speechtokenizer_num_layers is None or self.speechtokenizer_num_layers == 0:
                self.speechtokenizer_num_layers = 8  # Default fallback
            
            # Add tokens for each codebook value in each layer
            # Format: <audio_{layer}_{code}> where code goes from 1 to codebook_size
            for layer_idx in range(self.speechtokenizer_num_layers):
                for code_idx in range(1, self.speechtokenizer_codebook_size + 1):  # Codes from 1 to codebook_size
                    speechtokenizer_tokens.append(f"<audio_{layer_idx}_{code_idx}>")
            
            logging.info(f"Adding {len(speechtokenizer_tokens)} SpeechTokenizer tokens ({self.speechtokenizer_num_layers} layers × {self.speechtokenizer_codebook_size} codes)")
        
        # Add SQ-Codec tokens if using SQ-Codec
        if self.use_sqcodec:
            # Load config early to get num_quantizers and num_levels
            if audio_encoder_path:
                import json
                if os.path.isdir(audio_encoder_path):
                    config_path = os.path.join(audio_encoder_path, "config.json")
                else:
                    config_path = audio_encoder_path
                    audio_encoder_path = os.path.dirname(audio_encoder_path)
                
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            sq_config = json.load(f)
                        if "num_quantizers" in sq_config:
                            self.sqcodec_num_quantizers = sq_config["num_quantizers"]
                        elif "n_q" in sq_config:
                            self.sqcodec_num_quantizers = sq_config["n_q"]
                        if "num_levels" in sq_config:
                            self.sqcodec_num_levels = sq_config["num_levels"]
                        elif "codebook_size" in sq_config:
                            self.sqcodec_num_levels = sq_config["codebook_size"]
                        logging.info(f"Loaded SQ-Codec config: num_quantizers={self.sqcodec_num_quantizers}, num_levels={self.sqcodec_num_levels}")
                    except Exception as e:
                        logging.warning(f"Could not load SQ-Codec config from {config_path}: {e}, using defaults")
            
            # Use defaults if not set
            if self.sqcodec_num_levels is None:
                self.sqcodec_num_levels = 4096  # Default fallback
            if self.sqcodec_num_quantizers is None or self.sqcodec_num_quantizers == 0:
                self.sqcodec_num_quantizers = 4  # Default fallback (for SQ-SMA-16)
            
            # Add tokens for each quantization level in each quantizer
            # Format: <sqcodec_{quantizer}_{level}> where quantizer goes from 0 to num_quantizers-1, level from 0 to num_levels-1
            for quantizer_idx in range(self.sqcodec_num_quantizers):
                for level_idx in range(self.sqcodec_num_levels):
                    sqcodec_tokens.append(f"<sqcodec_{quantizer_idx}_{level_idx}>")
            
            logging.info(f"Adding {len(sqcodec_tokens)} SQ-Codec tokens ({self.sqcodec_num_quantizers} quantizers × {self.sqcodec_num_levels} levels)")
        
        # Combine with user-provided special tokens
        all_special_tokens = speechtokenizer_tokens + sqcodec_tokens
        if special_tokens:
            all_special_tokens = all_special_tokens + special_tokens
        
        if all_special_tokens:
            logging.info(f"Adding special tokens: {all_special_tokens}")
            num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': all_special_tokens})
            if num_added > 0:
                self.added_tokens = True
                logging.info(f"Added {num_added} special tokens to vocabulary")
        
        # Note: Token mapping for SpeechTokenizer will be built after LLM loading and embedding resize

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
        
        # Build SpeechTokenizer token mapping after embeddings are resized
        if self.use_speechtokenizer:
            self.speechtokenizer_token_map = {}
            for layer_idx in range(self.speechtokenizer_num_layers):
                for code_idx in range(1, self.speechtokenizer_codebook_size + 1):  # Codes from 1 to codebook_size
                    token_str = f"<audio_{layer_idx}_{code_idx}>"
                    token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                    # SpeechTokenizer codes are 0-indexed (0 to codebook_size-1)
                    # But our tokens are 1-indexed (1 to codebook_size)
                    # So we map: code_value -> token_id for (layer, code_value + 1)
                    self.speechtokenizer_token_map[(layer_idx, code_idx - 1)] = token_id
            
            logging.info(f"Built token mapping for {len(self.speechtokenizer_token_map)} (layer, code) pairs: {self.speechtokenizer_num_layers} layers × {self.speechtokenizer_codebook_size} codes")
        
        # Build SQ-Codec token mapping after embeddings are resized
        if self.use_sqcodec:
            self.sqcodec_token_map = {}
            for quantizer_idx in range(self.sqcodec_num_quantizers):
                for level_idx in range(self.sqcodec_num_levels):
                    token_str = f"<sqcodec_{quantizer_idx}_{level_idx}>"
                    token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                    # SQ-Codec codes are 0-indexed (0 to num_levels-1) for each quantizer
                    self.sqcodec_token_map[(quantizer_idx, level_idx)] = token_id
            
            logging.info(f"Built SQ-Codec token mapping for {len(self.sqcodec_token_map)} (quantizer, level) pairs: {self.sqcodec_num_quantizers} quantizers × {self.sqcodec_num_levels} levels")

        # Freeze LLM
        for param in self.llm_model.parameters():
            param.requires_grad = False
            
        # Apply LoRA
        if self.lora:
            # For discrete tokenizers (SpeechTokenizer, SQ-Codec), we MUST train embeddings/head to learn the new tokens
            # Also enable full training of embedding matrix and LM head
            modules_to_save = []
            if self.added_tokens or self.use_discrete_tokens:
                logging.info("Enabling training of embeddings and lm_head for new tokens via modules_to_save")
                modules_to_save = ["embed_tokens", "lm_head"]
            
            # If using SpeechTokenizer, also enable full training of embedding and head
            if self.use_speechtokenizer:
                logging.info("SpeechTokenizer mode: Enabling full training of embedding matrix and LM head")
                # Unfreeze embedding and LM head for SpeechTokenizer tokens
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
        # 2. Load Audio Encoder
        # ====================
        logging.info(f'Loading {self.audio_encoder_type} from {audio_encoder_path} (Source: {self.audio_encoder_source})')
        
        if self.audio_encoder_type == "whisper":
            self.audio_encoder = WhisperModel.from_pretrained(audio_encoder_path).encoder
            self.audio_dim = self.audio_encoder.config.d_model
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
        
        elif self.audio_encoder_type == "whisper_ner":
             # Initialize custom WhisperNER model
            from .whisper_ner import Whisper, ModelDimensions
            
            # Using Large-v3 dimensions by default
            dims = ModelDimensions(
                n_mels=128, 
                n_audio_ctx=1500, 
                n_audio_state=1280, 
                n_audio_head=20, 
                n_audio_layer=32, 
                n_vocab=51865, 
                n_text_ctx=448, 
                n_text_state=1280, 
                n_text_head=20, 
                n_text_layer=32
            )
            
            self.audio_encoder = Whisper(dims).encoder
            
            # Load weights if path is provided
            if audio_encoder_path and os.path.exists(audio_encoder_path):
                logging.info(f"Loading WhisperNER weights from {audio_encoder_path}")
                checkpoint = torch.load(audio_encoder_path, map_location="cpu")
                
                # Handle keys if needed (strip "model.encoder." prefix)
                new_state_dict = {}
                for k, v in checkpoint.items():
                    if k.startswith("model.encoder."):
                        new_key = k.replace("model.encoder.", "")
                        new_state_dict[new_key] = v

                        
                # Load with strict=False first to be safe, or True if you are sure
                try:
                    self.audio_encoder.load_state_dict(new_state_dict, strict=True)
                except RuntimeError as e:
                    logging.warning(f"Strict loading failed, trying strict=False: {e}")
                    self.audio_encoder.load_state_dict(new_state_dict, strict=False)

            self.audio_dim = 1280 # Large-v3 dimension
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

        elif self.audio_encoder_type == "wav2vec2":
            if self.audio_encoder_source == "fairseq":
                import fairseq
                models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([audio_encoder_path])
                self.audio_encoder = models[0]
                self.audio_dim = cfg.model.encoder_embed_dim
            else: # huggingface
                self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_path)
                self.audio_dim = self.audio_encoder.config.hidden_size
            
        elif self.audio_encoder_type == "wavlm":
            if self.audio_encoder_source in ["unil", "microsoft"]:
                from deepfense.models.modules.wavlm.wavlm import WavLM, WavLMConfig
                checkpoint = torch.load(audio_encoder_path)
                cfg = WavLMConfig(checkpoint["cfg"])
                self.audio_encoder = WavLM(cfg)
                self.audio_encoder.load_state_dict(checkpoint["model"], strict=False)
                self.audio_dim = cfg.encoder_embed_dim
            else: # huggingface
                self.audio_encoder = WavLMModel.from_pretrained(audio_encoder_path)
                self.audio_dim = self.audio_encoder.config.hidden_size
        
        elif self.audio_encoder_type == "wav2vec2-bert":
            self.audio_encoder = Wav2Vec2BertModel.from_pretrained(audio_encoder_path)
            self.audio_dim = self.audio_encoder.config.hidden_size
            self.processor = AutoFeatureExtractor.from_pretrained(audio_encoder_path)

        elif self.audio_encoder_type == "eat":
            from deepfense.models.frontends.eat import EATWrapper
            eat_config = {
                "source": "huggingface",
                "ckpt_path": audio_encoder_path,
                "freeze": freeze_audio_encoder
            }
            self.audio_encoder = EATWrapper(eat_config)
            # Try to infer dim
            if hasattr(self.audio_encoder.model, "config"):
                self.audio_dim = self.audio_encoder.model.config.embed_dim
            else:
                self.audio_dim = 1024 # EAT-large default
        
        elif self.audio_encoder_type == "speechtokenizer":
            # Load SpeechTokenizer model
            try:
                from speechtokenizer import SpeechTokenizer
            except ImportError:
                try:
                    from speechtokenizer.speechtokenizer import SpeechTokenizer
                except ImportError:
                    raise ImportError("SpeechTokenizer not found. Please install: pip install speechtokenizer")
            
            # Determine config_path and ckpt_path
            # audio_encoder_path can be a directory or config file path
            if os.path.isdir(audio_encoder_path):
                config_path = os.path.join(audio_encoder_path, "config.json")
                # Look for .pt file in the directory
                pt_files = [f for f in os.listdir(audio_encoder_path) if f.endswith('.pt')]
                if pt_files:
                    ckpt_path = os.path.join(audio_encoder_path, pt_files[0])
                else:
                    raise FileNotFoundError(f"No .pt file found in {audio_encoder_path}")
            else:
                # Assume it's a config file path
                config_path = audio_encoder_path
                audio_encoder_dir = os.path.dirname(audio_encoder_path)
                pt_files = [f for f in os.listdir(audio_encoder_dir) if f.endswith('.pt')]
                if pt_files:
                    ckpt_path = os.path.join(audio_encoder_dir, pt_files[0])
                else:
                    raise FileNotFoundError(f"No .pt file found in {audio_encoder_dir}")
            
            logging.info(f"Loading SpeechTokenizer from config: {config_path}, checkpoint: {ckpt_path}")
            self.audio_encoder = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
            # Always freeze SpeechTokenizer in eval mode
            self.audio_encoder.eval()
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            
            # Get codebook info from loaded model (should already be set from config)
            if hasattr(self.audio_encoder, "n_q") and hasattr(self.audio_encoder, "codebook_size"):
                self.speechtokenizer_num_layers = self.audio_encoder.n_q
                self.speechtokenizer_codebook_size = self.audio_encoder.codebook_size
            elif hasattr(self.audio_encoder, "config"):
                # Try to get from config
                if hasattr(self.audio_encoder.config, "n_q"):
                    self.speechtokenizer_num_layers = self.audio_encoder.config.n_q
                if hasattr(self.audio_encoder.config, "codebook_size"):
                    self.speechtokenizer_codebook_size = self.audio_encoder.config.codebook_size
            
            logging.info(f"SpeechTokenizer: {self.speechtokenizer_num_layers} layers, codebook size {self.speechtokenizer_codebook_size}")
            
            # Token mapping was already built before LLM loading
            # Verify the mapping is correct
            if not hasattr(self, 'speechtokenizer_token_map') or len(self.speechtokenizer_token_map) == 0:
                # Rebuild if somehow missing
                self.speechtokenizer_token_map = {}
                for layer_idx in range(self.speechtokenizer_num_layers):
                    for code_idx in range(1, self.speechtokenizer_codebook_size + 1):
                        token_str = f"<audio_{layer_idx}_{code_idx}>"
                        token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                        self.speechtokenizer_token_map[(layer_idx, code_idx - 1)] = token_id
                logging.info(f"Rebuilt token mapping for {len(self.speechtokenizer_token_map)} (layer, code) pairs")
            
            # SpeechTokenizer doesn't have a continuous embedding dimension
            # We'll use discrete tokens instead
            self.audio_dim = None
        
        elif self.audio_encoder_type == "sqcodec":
            # Load SQ-Codec model
            # typing.Self should already be patched at module level
            try:
                import sq_codec
            except ImportError:
                try:
                    import sqcodec as sq_codec
                except ImportError as e:
                    if "Self" in str(e):
                        raise ImportError(
                            "SQ-Codec requires Python 3.11+ or typing_extensions. "
                            "Install typing_extensions: pip install typing_extensions"
                        )
                    raise ImportError("SQ-Codec not found. Please install: pip install sq-codec")
            
            # Determine model path/name
            # audio_encoder_path can be a directory, model name (e.g., "SQ-SMA-16"), or model path
            if os.path.isdir(audio_encoder_path):
                # Directory containing model files
                model_path = audio_encoder_path
            else:
                # Could be a model name (e.g., "SQ-SMA-16") or file path
                model_path = audio_encoder_path
            
            logging.info(f"Loading SQ-Codec from {model_path}")
            
            # Try to load using sq_codec API
            # sq_codec.get_model() for model names, or from_pretrained for paths
            if hasattr(sq_codec, "get_model"):
                # Try as model name first (e.g., "SQ-SMA-16", "6kbps", etc.)
                try:
                    self.audio_encoder = sq_codec.get_model(model_path)
                except:
                    # If that fails, try as path
                    if os.path.exists(model_path) or os.path.isdir(model_path):
                        if hasattr(sq_codec, "SQCodec") and hasattr(sq_codec.SQCodec, "from_pretrained"):
                            self.audio_encoder = sq_codec.SQCodec.from_pretrained(model_path)
                        else:
                            # Fallback: try loading checkpoint directly
                            raise ValueError(f"Could not load SQ-Codec from {model_path}. Check if model name is correct or path exists.")
                    else:
                        raise ValueError(f"SQ-Codec model '{model_path}' not found. Available models: {sq_codec.list_models() if hasattr(sq_codec, 'list_models') else 'N/A'}")
            elif hasattr(sq_codec, "SQCodec"):
                # Use SQCodec class directly
                if os.path.exists(model_path) or os.path.isdir(model_path):
                    self.audio_encoder = sq_codec.SQCodec.from_pretrained(model_path)
                else:
                    raise ValueError(f"SQ-Codec path not found: {model_path}")
            else:
                raise ImportError("SQ-Codec module structure not recognized. Please check sq-codec installation.")
            
            # Always freeze SQ-Codec in eval mode
            # SQ-Codec might have network attribute
            if hasattr(self.audio_encoder, "network"):
                self.audio_encoder.network.eval()
                for param in self.audio_encoder.network.parameters():
                    param.requires_grad = False
            else:
                self.audio_encoder.eval()
                for param in self.audio_encoder.parameters():
                    param.requires_grad = False
            
            # Get codebook info from loaded model
            # SQ-Codec might have network attribute that contains the actual model
            codec_model = self.audio_encoder
            if hasattr(self.audio_encoder, "network"):
                codec_model = self.audio_encoder.network
            
            if hasattr(codec_model, "num_quantizers"):
                self.sqcodec_num_quantizers = codec_model.num_quantizers
            elif hasattr(codec_model, "n_q"):
                self.sqcodec_num_quantizers = codec_model.n_q
            elif hasattr(self.audio_encoder, "config"):
                if hasattr(self.audio_encoder.config, "num_quantizers"):
                    self.sqcodec_num_quantizers = self.audio_encoder.config.num_quantizers
                elif hasattr(self.audio_encoder.config, "n_q"):
                    self.sqcodec_num_quantizers = self.audio_encoder.config.n_q
            
            if hasattr(codec_model, "num_levels"):
                self.sqcodec_num_levels = codec_model.num_levels
            elif hasattr(codec_model, "codebook_size"):
                self.sqcodec_num_levels = codec_model.codebook_size
            elif hasattr(self.audio_encoder, "config"):
                if hasattr(self.audio_encoder.config, "num_levels"):
                    self.sqcodec_num_levels = self.audio_encoder.config.num_levels
                elif hasattr(self.audio_encoder.config, "codebook_size"):
                    self.sqcodec_num_levels = self.audio_encoder.config.codebook_size
            
            logging.info(f"SQ-Codec: num_quantizers={self.sqcodec_num_quantizers}, num_levels={self.sqcodec_num_levels}")
            
            # Token mapping was already built before LLM loading
            # Verify the mapping is correct
            if not hasattr(self, 'sqcodec_token_map') or len(self.sqcodec_token_map) == 0:
                # Rebuild if somehow missing
                self.sqcodec_token_map = {}
                for quantizer_idx in range(self.sqcodec_num_quantizers):
                    for level_idx in range(self.sqcodec_num_levels):
                        token_str = f"<sqcodec_{quantizer_idx}_{level_idx}>"
                        token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                        self.sqcodec_token_map[(quantizer_idx, level_idx)] = token_id
                logging.info(f"Rebuilt SQ-Codec token mapping for {len(self.sqcodec_token_map)} (quantizer, level) pairs")
            
            # SQ-Codec doesn't have a continuous embedding dimension
            # We'll use discrete tokens instead
            self.audio_dim = None
            
        else:
            raise ValueError(f"Unknown audio encoder type: {self.audio_encoder_type}")

        if freeze_audio_encoder and self.audio_encoder_type not in ["eat", "speechtokenizer", "sqcodec"]:
            # EAT handles freeze internally, SpeechTokenizer and SQ-Codec are always frozen (handled above)
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
        
        # SpeechTokenizer and SQ-Codec are always frozen (handled during loading above)

        # Initialize intermediate layer weights if enabled
        if self.use_intermediate_layers:
            num_layers = None
            
            # Get number of layers from encoder
            if self.audio_encoder_type in ["whisper", "whisper_ner"]:
                if hasattr(self.audio_encoder, "layers"):
                    num_layers = len(self.audio_encoder.layers)
                elif hasattr(self.audio_encoder, "config") and hasattr(self.audio_encoder.config, "encoder_layers"):
                    num_layers = self.audio_encoder.config.encoder_layers
                elif hasattr(self.audio_encoder, "config") and hasattr(self.audio_encoder.config, "num_hidden_layers"):
                    num_layers = self.audio_encoder.config.num_hidden_layers
                else:
                    # Default for whisper large-v3
                    num_layers = 32
                    logging.warning(f"Could not determine number of layers for {self.audio_encoder_type}, using default {num_layers}")
            
            elif self.audio_encoder_type == "wav2vec2":
                if self.audio_encoder_source == "fairseq":
                    if hasattr(self.audio_encoder, "encoder") and hasattr(self.audio_encoder.encoder, "layers"):
                        num_layers = len(self.audio_encoder.encoder.layers)
                    else:
                        # Try to get from config
                        num_layers = getattr(self.audio_encoder, "num_layers", None)
                        if num_layers is None:
                            num_layers = 24  # Default for wav2vec2-base
                            logging.warning(f"Could not determine number of layers for fairseq wav2vec2, using default {num_layers}")
                else:  # huggingface
                    if hasattr(self.audio_encoder, "encoder") and hasattr(self.audio_encoder.encoder, "layers"):
                        num_layers = len(self.audio_encoder.encoder.layers)
                    elif hasattr(self.audio_encoder, "config") and hasattr(self.audio_encoder.config, "num_hidden_layers"):
                        num_layers = self.audio_encoder.config.num_hidden_layers
                    elif hasattr(self.audio_encoder, "config") and hasattr(self.audio_encoder.config, "encoder_layers"):
                        num_layers = self.audio_encoder.config.encoder_layers
                    else:
                        num_layers = 25  # Default for wav2vec2-base
                        logging.warning(f"Could not determine number of layers for huggingface wav2vec2, using default {num_layers}")
            
            if num_layers is None:
                logging.warning(f"Could not determine number of layers for {self.audio_encoder_type}, disabling intermediate layers")
                self.use_intermediate_layers = False
            else:
                # Determine which layers to use
                if self.selected_layers is not None:
                    # Use specified layers
                    layer_indices = [idx for idx in self.selected_layers if 0 <= idx < num_layers]
                    if len(layer_indices) == 0:
                        logging.warning(f"Selected layers {self.selected_layers} are all out of range [0, {num_layers}), using all layers")
                        layer_indices = list(range(num_layers))
                else:
                    # Use all layers
                    layer_indices = list(range(num_layers))
                
                num_selected_layers = len(layer_indices)
                self.selected_layers = layer_indices
                
                # Initialize weights
                if layer_weights is not None:
                    if len(layer_weights) != num_selected_layers:
                        logging.warning(f"Provided {len(layer_weights)} weights but {num_selected_layers} layers selected. Using uniform weights.")
                        weights = torch.ones(num_selected_layers, dtype=torch.float32)
                    else:
                        weights = torch.tensor(layer_weights, dtype=torch.float32)
                        # Ensure positive
                        weights = torch.clamp(weights, min=1e-6)
                else:
                    # Uniform weights
                    weights = torch.ones(num_selected_layers, dtype=torch.float32)
                
                # Normalize to sum to 1
                weights = weights / weights.sum()
                
                # Make trainable parameters (always trainable, even if audio encoder is frozen)
                self.layer_weights = nn.Parameter(weights)
                self.layer_weights.requires_grad = True  # Ensure they're trainable
                
                logging.info(f"Initialized intermediate layer weights for {num_selected_layers} layers (indices: {layer_indices})")
                logging.info(f"Initial weights (sum={self.layer_weights.sum().item():.4f}): {self.layer_weights.data.tolist()}")

        # Audio Projector (only for continuous embeddings, not for discrete tokenizers)
        if not self.use_discrete_tokens:
            # Ensure projection is in correct dtype
            self.speech_proj = nn.Linear(self.audio_dim, self.llm_model.config.hidden_size)
            if freeze_adapter:
                for param in self.speech_proj.parameters():
                    param.requires_grad = False
                self.speech_proj.eval()
        else:
            # No projector needed for SpeechTokenizer (we use discrete tokens)
            self.speech_proj = None

    def get_layer_weights(self):
        """
        Get current normalized layer weights (summing to 2).
        Returns None if intermediate layers are not enabled.
        """
        if not self.use_intermediate_layers or self.layer_weights is None:
            return None
        # Return normalized weights (softmax * 2)
        normalized = F.softmax(self.layer_weights, dim=0) * 2.0
        return {
            'raw_weights': self.layer_weights.data.cpu().tolist(),
            'normalized_weights': normalized.data.cpu().tolist(),
            'sum': normalized.sum().item(),
            'selected_layers': self.selected_layers
        }

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """
        if self.audio_encoder_type in ["whisper", "whisper_ner"]:
            # Whisper: 16kHz -> 10ms stride (160 samples) -> Encoder
            # Whisper NER shares the same architecture
            
            # Note: For custom Whisper implementation (whisper_ner), 
            # we might need to manually calculate if the method doesn't exist
            if hasattr(self.audio_encoder, "_get_feat_extract_output_lengths"):
                 feat_lens = (input_lengths / 160).ceil().long()
                 return self.audio_encoder._get_feat_extract_output_lengths(feat_lens)
            else:
                # Fallback calculation for Whisper convolution (Conv1d stride 1, then Conv1d stride 2)
                # Input: [B, 80/128, T]
                # Conv1: stride 1, padding 1 -> T
                # Conv2: stride 2, padding 1 -> (T + 2*1 - 3)/2 + 1 = (T-1)/2 + 1 = floor((T-1)/2) + 1
                # Usually simplified to T/2
                feat_lens = (input_lengths / 160).ceil().long()
                return (feat_lens + 1) // 2 
            
        elif self.audio_encoder_type == "wav2vec2-bert":
            # W2V-BERT 2.0 uses 160-dim fbank features extracted every 10ms (160 samples)
            # 1. Map samples to frames (stride 160)
            feat_lens = (input_lengths / 160).ceil().long()
            
            # 2. Map frames to encoder output (usually 4x subsampling)
            return self.audio_encoder._get_feat_extract_output_lengths(feat_lens)

        elif self.audio_encoder_type == "eat":
            feat_lens = (input_lengths / 160).floor().long()
            t_patches = (feat_lens / 16).floor().long()
            # 8 freq patches + 1 CLS
            return t_patches * 8 + 1
        
        elif self.audio_encoder_type == "sqcodec":
            # SQ-Codec typically has 320 sample stride (similar to SpeechTokenizer)
            return (input_lengths / 320).ceil().long()
            
        elif hasattr(self.audio_encoder, "_get_feat_extract_output_lengths"):
            return self.audio_encoder._get_feat_extract_output_lengths(input_lengths)
            
        else:
            # Fallback for Fairseq/WavLM custom models
            # Standard Wav2Vec2/WavLM stride is 320
            return (input_lengths / 320).ceil().long()

    def _process_audio(self, raw_wav, audio_attention_mask=None):
        """
        Extract features from raw audio based on encoder type.
        Returns:
            hidden_states: [B, T_feat, D]
            speech_lens: [B] (Valid lengths of features)
        """
        device = raw_wav.device
        
        # Calculate raw lengths
        if audio_attention_mask is not None:
            input_lengths = audio_attention_mask.sum(dim=1)
        else:
            input_lengths = torch.tensor([raw_wav.shape[1]] * raw_wav.shape[0], device=device)
            
        speech_lens = self._get_feat_extract_output_lengths(input_lengths)
        
        if self.audio_encoder_type in ["whisper", "whisper_ner"]:
            # Whisper expects log-mel spectrograms of specific length
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
            if self.use_intermediate_layers:
                # Extract intermediate layers
                outputs = self.audio_encoder(input_features, output_hidden_states=True)
                
                if isinstance(outputs, torch.Tensor):
                    # Custom WhisperNER might return tensor directly
                    hidden_states = outputs
                else:
                    # Get all hidden states
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        all_hidden_states = outputs.hidden_states  # Tuple of [B, T, D] for each layer
                    elif hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
                        all_hidden_states = outputs.encoder_hidden_states
                    else:
                        # Fallback to last hidden state
                        all_hidden_states = (outputs.last_hidden_state,) if hasattr(outputs, 'last_hidden_state') else (outputs,)
                    
                    # Select and weight layers
                    # Normalize weights to ensure they sum to 2 (with softmax-like normalization)
                    weights = F.softmax(self.layer_weights, dim=0) * 2.0
                    
                    # Get selected layers
                    selected_hidden_states = [all_hidden_states[idx] for idx in self.selected_layers if idx < len(all_hidden_states)]
                    
                    if len(selected_hidden_states) == 0:
                        # Fallback to last layer
                        hidden_states = all_hidden_states[-1] if len(all_hidden_states) > 0 else outputs.last_hidden_state
                    else:
                        # Weighted combination
                        # Ensure we have the right number of weights
                        num_layers = len(selected_hidden_states)
                        if len(weights) != num_layers:
                            # Re-normalize if mismatch
                            weights = F.softmax(self.layer_weights[:num_layers], dim=0) * 2.0
                        
                        # Stack and weight: [num_layers, B, T, D]
                        stacked = torch.stack(selected_hidden_states, dim=0)
                        # Apply weights: [num_layers, B, T, D] * [num_layers, 1, 1, 1] -> [num_layers, B, T, D]
                        weights_expanded = weights.view(-1, 1, 1, 1)
                        weighted = stacked * weights_expanded
                        # Sum over layer dimension: [B, T, D]
                        hidden_states = weighted.sum(dim=0)
            else:
                # Standard: use only last layer
                outputs = self.audio_encoder(input_features)
                
                if isinstance(outputs, torch.Tensor):
                    hidden_states = outputs
                else:
                    hidden_states = outputs.last_hidden_state # [B, 1500, D]
            
            # Clamp speech_lens to 1500 just in case
            speech_lens = torch.clamp(speech_lens, max=1500)
            
            return hidden_states, speech_lens
            
        elif self.audio_encoder_type in ["wav2vec2", "wavlm"]:
            # Wav2Vec2/WavLM expect raw audio
            
            # Handle different sources
            if self.audio_encoder_source == "fairseq":
                # Fairseq Wav2Vec2
                # Expects input: [B, T]
                # Returns dict with 'x'
                
                # IMPORTANT: Fairseq Wav2Vec2 expects Float tensor, not BFloat16/Half
                if raw_wav.dtype != torch.float32:
                    raw_wav_input = raw_wav.to(torch.float32)
                else:
                    raw_wav_input = raw_wav
                
                if self.use_intermediate_layers and self.audio_encoder_type == "wav2vec2":
                    # Extract intermediate layers from fairseq
                    # Fairseq models support ret_layer_results parameter
                    try:
                        if hasattr(self.audio_encoder, "extract_features"):
                            # Try to extract with layer results
                            features, layer_results = self.audio_encoder.extract_features(
                                raw_wav_input, mask=False, ret_layer_results=True
                            )
                            # layer_results is a list of tuples (x, padding_mask) for each layer
                            all_hidden_states = [layer_res[0] for layer_res in layer_results] + [features]
                        else:
                            # Fallback: try to get from encoder directly
                            # Some fairseq models store layers in encoder.layers
                            if hasattr(self.audio_encoder, "encoder") and hasattr(self.audio_encoder.encoder, "layers"):
                                # Manually forward through layers (more complex, fallback to last layer)
                                logging.warning("Fairseq wav2vec2 intermediate layers extraction not fully supported, using last layer")
                                res = self.audio_encoder(raw_wav_input, mask=False, features_only=True)
                                hidden_states = res["x"]
                                return hidden_states, speech_lens
                            else:
                                res = self.audio_encoder(raw_wav_input, mask=False, features_only=True)
                                hidden_states = res["x"]
                                return hidden_states, speech_lens
                        
                        # Select and weight layers
                        weights = F.softmax(self.layer_weights, dim=0) * 2.0
                        selected_hidden_states = [all_hidden_states[idx] for idx in self.selected_layers if idx < len(all_hidden_states)]
                        
                        if len(selected_hidden_states) == 0:
                            hidden_states = features
                        else:
                            num_layers = len(selected_hidden_states)
                            if len(weights) != num_layers:
                                weights = F.softmax(self.layer_weights[:num_layers], dim=0) * 2.0
                            
                            stacked = torch.stack(selected_hidden_states, dim=0)
                            weights_expanded = weights.view(-1, 1, 1, 1)
                            weighted = stacked * weights_expanded
                            hidden_states = weighted.sum(dim=0)
                    except Exception as e:
                        logging.warning(f"Failed to extract intermediate layers from fairseq wav2vec2: {e}. Using last layer.")
                        res = self.audio_encoder(raw_wav_input, mask=False, features_only=True)
                        hidden_states = res["x"]
                else:
                    res = self.audio_encoder(raw_wav_input, mask=False, features_only=True)
                    hidden_states = res["x"]
                
                return hidden_states, speech_lens
                
            elif self.audio_encoder_source in ["unil", "microsoft"]:
                # Microsoft WavLM
                # Expects input: [B, T]
                # Returns x (features)
                
                if raw_wav.dtype != torch.float32:
                    raw_wav_input = raw_wav.to(torch.float32)
                else:
                    raw_wav_input = raw_wav
                
                if self.use_intermediate_layers:
                    # Extract with layer results
                    features, layer_results = self.audio_encoder.extract_features(
                        raw_wav_input, mask=False, ret_layer_results=True
                    )
                    all_hidden_states = [layer_res[0] for layer_res in layer_results] + [features]
                    
                    # Select and weight layers
                    weights = F.softmax(self.layer_weights, dim=0) * 2.0
                    selected_hidden_states = [all_hidden_states[idx] for idx in self.selected_layers if idx < len(all_hidden_states)]
                    
                    if len(selected_hidden_states) == 0:
                        hidden_states = features
                    else:
                        num_layers = len(selected_hidden_states)
                        if len(weights) != num_layers:
                            weights = F.softmax(self.layer_weights[:num_layers], dim=0) * 2.0
                        
                        stacked = torch.stack(selected_hidden_states, dim=0)
                        weights_expanded = weights.view(-1, 1, 1, 1)
                        weighted = stacked * weights_expanded
                        hidden_states = weighted.sum(dim=0)
                else:
                    hidden_states = self.audio_encoder.extract_features(
                        raw_wav_input, mask=False, ret_layer_results=False
                    )[0]
                
                return hidden_states, speech_lens
                
            else: # huggingface
                if self.use_intermediate_layers and self.audio_encoder_type == "wav2vec2":
                    # Extract intermediate layers
                    outputs = self.audio_encoder(
                        raw_wav, 
                        attention_mask=audio_attention_mask,
                        output_hidden_states=True
                    )
                    
                    # Get all hidden states
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        all_hidden_states = outputs.hidden_states
                    else:
                        # Fallback
                        all_hidden_states = (outputs.last_hidden_state,)
                    
                    # Select and weight layers
                    weights = F.softmax(self.layer_weights, dim=0) * 2.0
                    selected_hidden_states = [all_hidden_states[idx] for idx in self.selected_layers if idx < len(all_hidden_states)]
                    
                    if len(selected_hidden_states) == 0:
                        hidden_states = outputs.last_hidden_state
                    else:
                        num_layers = len(selected_hidden_states)
                        if len(weights) != num_layers:
                            weights = F.softmax(self.layer_weights[:num_layers], dim=0) * 2.0
                        
                        stacked = torch.stack(selected_hidden_states, dim=0)
                        weights_expanded = weights.view(-1, 1, 1, 1)
                        weighted = stacked * weights_expanded
                        hidden_states = weighted.sum(dim=0)
                else:
                    outputs = self.audio_encoder(
                        raw_wav, 
                        attention_mask=audio_attention_mask
                    )
                    hidden_states = outputs.last_hidden_state
                
                return hidden_states, speech_lens
                
        elif self.audio_encoder_type == "wav2vec2-bert":
            # W2V-BERT requires feature extraction (fbank) via processor
            with torch.no_grad():
                raw_wav_cpu = raw_wav.cpu().numpy()
                # Processor expects list of arrays
                inputs = self.processor(
                    [w for w in raw_wav_cpu], 
                    sampling_rate=16000, 
                    return_tensors="pt",
                    padding="longest"
                )
                input_features = inputs.input_features.to(device)
                attention_mask = inputs.attention_mask.to(device)
            
            outputs = self.audio_encoder(
                input_features=input_features, 
                attention_mask=attention_mask
            )
            return outputs.last_hidden_state, speech_lens

        elif self.audio_encoder_type == "eat":
            # Padding management for patch-based processing
            # 1 frame = 10ms = 160 samples
            # Patch time dim = 16 frames
            # Block size = 16 * 160 = 2560 samples
            
            if raw_wav.dtype != torch.float32:
                raw_wav_input = raw_wav.to(torch.float32)
            else:
                raw_wav_input = raw_wav

            hidden_states = self.audio_encoder(raw_wav_input)
            
            return hidden_states, speech_lens
        
        elif self.audio_encoder_type == "speechtokenizer":
            # SpeechTokenizer returns discrete codes
            # Input: [B, T] raw audio
            # Output: codes [n_q, B, T_code] where n_q is number of quantizer layers
            with torch.no_grad():
                if raw_wav.dtype != torch.float32:
                    raw_wav_input = raw_wav.to(torch.float32)
                else:
                    raw_wav_input = raw_wav
                
                # SpeechTokenizer.encode returns codes: [n_q, B, T]
                codes = self.audio_encoder.encode(raw_wav_input.unsqueeze(1))  # [n_q, B, T]
                codes = codes[:self.speechtokenizer_num_layers, :, :]
                # Transpose to [B, n_q, T] for easier handling
                codes = codes.transpose(0, 1)  # [B, n_q, T]

            # Calculate output lengths based on actual code length
            # SpeechTokenizer typically has 320 sample stride, but we use actual code length
            batch_size = codes.shape[0]
            n_q = codes.shape[1]
            max_code_len = codes.shape[2]
            
            # Calculate valid lengths from input audio lengths
            # SpeechTokenizer stride is typically 320 samples (20ms at 16kHz)
            # But we should use the actual code length if available
            if audio_attention_mask is not None:
                # Use actual input lengths to calculate code lengths
                input_lengths = audio_attention_mask.sum(dim=1)
                # SpeechTokenizer stride: typically 320 samples per code
                speech_lens = (input_lengths / 320).ceil().long()
                # Clamp to max_code_len to avoid out-of-bounds
                speech_lens = torch.clamp(speech_lens, max=max_code_len)
            else:
                # If no mask, assume all codes are valid (up to max_code_len)
                # But we should still calculate based on input length
                input_lengths = torch.tensor([raw_wav.shape[1]] * batch_size, device=device)
                speech_lens = (input_lengths / 320).ceil().long()
                speech_lens = torch.clamp(speech_lens, max=max_code_len)
            
            return codes, speech_lens
        
        elif self.audio_encoder_type == "sqcodec":
            # SQ-Codec returns discrete codes
            # Input: [B, T] raw audio
            # Output: codes [n_q, B, T] where n_q is number of quantizers (similar to SpeechTokenizer)
            with torch.no_grad():
                if raw_wav.dtype != torch.float32:
                    raw_wav_input = raw_wav.to(torch.float32)
                else:
                    raw_wav_input = raw_wav
                
                # SQ-Codec encoding
                # SQ-Codec might have encode_audio method or encode method
                if hasattr(self.audio_encoder, "encode_audio"):
                    # sq_codec.get_model() returns object with encode_audio method
                    result = self.audio_encoder.encode_audio(raw_wav_input)
                    if isinstance(result, tuple):
                        # Returns (q_feature, indices) - we want indices
                        codes = result[1]  # indices: [n_q, B, T] or [B, n_q, T]
                    else:
                        codes = result
                elif hasattr(self.audio_encoder, "encode"):
                    codes = self.audio_encoder.encode(raw_wav_input)
                else:
                    # Try calling the model directly
                    codes = self.audio_encoder(raw_wav_input)
                    if isinstance(codes, tuple):
                        codes = codes[1]  # Assume codes are second element
                
                # Ensure codes are [B, n_q, T] format
                if codes.dim() == 3:
                    if codes.shape[0] == self.sqcodec_num_quantizers:
                        # [n_q, B, T] -> transpose to [B, n_q, T]
                        codes = codes.transpose(0, 1)
                    elif codes.shape[1] == self.sqcodec_num_quantizers:
                        # Already [B, n_q, T]
                        pass
                    else:
                        # Might be [B, T, n_q] -> transpose to [B, n_q, T]
                        if codes.shape[2] == self.sqcodec_num_quantizers:
                            codes = codes.transpose(1, 2)  # [B, n_q, T]
                        else:
                            raise ValueError(f"Unexpected SQ-Codec output shape: {codes.shape}, expected quantizers dimension")
                elif codes.dim() == 2:
                    # Single quantizer case: [B, T] -> expand to [B, 1, T]
                    if self.sqcodec_num_quantizers == 1:
                        codes = codes.unsqueeze(1)  # [B, 1, T]
                    else:
                        raise ValueError(f"SQ-Codec has {self.sqcodec_num_quantizers} quantizers but got 2D codes: {codes.shape}")
                else:
                    raise ValueError(f"Unexpected SQ-Codec output shape: {codes.shape}")
                
                # Slice to requested number of quantizers if needed
                if codes.shape[1] > self.sqcodec_num_quantizers:
                    codes = codes[:, :self.sqcodec_num_quantizers, :]
            
            # Calculate output lengths
            batch_size = codes.shape[0]
            max_code_len = codes.shape[2]
            
            if audio_attention_mask is not None:
                input_lengths = audio_attention_mask.sum(dim=1)
                # SQ-Codec typically has similar stride to SpeechTokenizer (320 samples)
                speech_lens = (input_lengths / 320).ceil().long()
                speech_lens = torch.clamp(speech_lens, max=max_code_len)
            else:
                input_lengths = torch.tensor([raw_wav.shape[1]] * batch_size, device=device)
                speech_lens = (input_lengths / 320).ceil().long()
                speech_lens = torch.clamp(speech_lens, max=max_code_len)
            
            return codes, speech_lens
            
        return None, None

    def encode_speech(self, raw_wav, audio_attention_mask=None):
        """
        Encode speech to either continuous embeddings or discrete tokens.
        Returns:
            - For continuous encoders: (embeddings, lengths) where embeddings are [B, T, D]
            - For SpeechTokenizer: (codes, lengths) where codes are [B, n_q, T] discrete token IDs
            - For SQ-Codec: (codes, lengths) where codes are [B, n_q, T] discrete token IDs (n_q quantizers)
        """
        if self.use_discrete_tokens:
            # Return discrete codes directly
            codes, speech_lens = self._process_audio(raw_wav, audio_attention_mask)
            return codes, speech_lens
        else:
            # Use continuous embeddings with projection
            with self.maybe_autocast():
                speech_embeds, speech_lens = self._process_audio(raw_wav, audio_attention_mask)
                speech_embeds = self.speech_proj(speech_embeds)
                return speech_embeds, speech_lens

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

    def prompt_wrap(self, speech_embeds, speech_lens, prompt, multi_prompt=False, padding_side="right", prefix_embeds=None):
        """
        Wraps speech embeddings/tokens with text prompts.
        For SpeechTokenizer: converts discrete codes to token IDs and concatenates with text tokens.
        For SQ-Codec: converts discrete codes to token IDs and concatenates with text tokens.
        For continuous encoders: concatenates embeddings with text embeddings.
        """
        batch_size = speech_embeds.shape[0]
        device = speech_embeds.device
        
        if self.use_discrete_tokens:
            # Handle discrete tokens from SpeechTokenizer or SQ-Codec
            return self._prompt_wrap_tokens(speech_embeds, speech_lens, prompt, multi_prompt, padding_side, prefix_embeds)
        else:
            # Handle continuous embeddings
            return self._prompt_wrap_embeds(speech_embeds, speech_lens, prompt, multi_prompt, padding_side, prefix_embeds)
    
    def _prompt_wrap_tokens(self, speech_codes, speech_lens, prompt, multi_prompt=False, padding_side="right", prefix_embeds=None):
        """
        Wraps discrete codes (SpeechTokenizer or SQ-Codec) with text prompts.
        For SpeechTokenizer: speech_codes is [B, n_q, T] discrete code values
        For SQ-Codec: speech_codes is [B, n_q, T] discrete code values (n_q quantizers)
        Returns: (input_ids, attention_mask) instead of embeddings
        """
        batch_size = speech_codes.shape[0]
        device = speech_codes.device
        
        if self.use_speechtokenizer:
            n_q = speech_codes.shape[1]  # Number of quantizer layers
        elif self.use_sqcodec:
            n_q = speech_codes.shape[1]  # Number of quantizers for SQ-Codec
        else:
            n_q = 1
        
        input_ids_list = []
        attn_mask_list = []
        
        if isinstance(prompts := prompt, str):
            prompts = [prompts] * batch_size

        for i, prompt in enumerate(prompts):
            p_before = ""
            p_after = ""
            
            # Try to apply chat template if available
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
            
            # Tokenize text parts
            tokens_before = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            tokens_after = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)
            
            # Convert discrete codes to token IDs
            # Get valid length (removes padding)
            valid_len = speech_lens[i].item()
            
            if self.use_speechtokenizer:
                # SpeechTokenizer: codes are [B, n_q, T]
                # Clamp to actual code length to avoid out-of-bounds
                actual_code_len = speech_codes.shape[2]
                valid_len = min(valid_len, actual_code_len)
                
                # Slice codes to valid length (removes padded codes)
                cur_codes = speech_codes[i, :, :valid_len]  # [n_q, valid_len]
                
                # Convert codes to token IDs: for each time step, we have n_q codes
                # We'll interleave them: layer_0_code, layer_1_code, ..., layer_n_q-1_code, layer_0_code+1, ...
                audio_token_ids = []
                for t in range(valid_len):
                    for layer_idx in range(n_q):
                        code_value = cur_codes[layer_idx, t].item()
                        
                        # Validate code value is within expected range
                        if code_value < 0 or code_value >= self.speechtokenizer_codebook_size:
                            logging.warning(f"Invalid code value {code_value} for layer {layer_idx} at time {t} (expected 0-{self.speechtokenizer_codebook_size-1}), skipping")
                            continue
                        
                        # Map (layer_idx, code_value) to token ID
                        token_id = self.speechtokenizer_token_map.get((layer_idx, code_value))
                        if token_id is None:
                            # Fallback: construct token string and get ID
                            token_str = f"<audio_{layer_idx}_{code_value + 1}>"
                            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                            if token_id == self.tokenizer.unk_token_id:
                                logging.warning(f"Token '{token_str}' not found in vocabulary, skipping")
                                continue
                        
                        audio_token_ids.append(token_id)
            
            elif self.use_sqcodec:
                # SQ-Codec: codes are [B, n_q, T] (similar to SpeechTokenizer)
                # Clamp to actual code length to avoid out-of-bounds
                actual_code_len = speech_codes.shape[2]
                valid_len = min(valid_len, actual_code_len)
                
                # Slice codes to valid length (removes padded codes)
                cur_codes = speech_codes[i, :, :valid_len]  # [n_q, valid_len]
                
                # Convert codes to token IDs: for each time step, we have n_q codes
                # We'll interleave them: quantizer_0_code, quantizer_1_code, ..., quantizer_n_q-1_code, quantizer_0_code+1, ...
                audio_token_ids = []
                for t in range(valid_len):
                    for quantizer_idx in range(n_q):
                        code_value = cur_codes[quantizer_idx, t].item()
                        
                        # Validate code value is within expected range
                        if code_value < 0 or code_value >= self.sqcodec_num_levels:
                            logging.warning(f"Invalid SQ-Codec code value {code_value} for quantizer {quantizer_idx} at time {t} (expected 0-{self.sqcodec_num_levels-1}), skipping")
                            continue
                        
                        # Map (quantizer_idx, code_value) to token ID
                        token_id = self.sqcodec_token_map.get((quantizer_idx, code_value))
                        if token_id is None:
                            # Fallback: construct token string and get ID
                            token_str = f"<sqcodec_{quantizer_idx}_{code_value}>"
                            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                            if token_id == self.tokenizer.unk_token_id:
                                logging.warning(f"Token '{token_str}' not found in vocabulary, skipping")
                                continue
                        
                        audio_token_ids.append(token_id)
            
            audio_token_ids = torch.tensor(audio_token_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T_audio]
            
            # Combine: [before_tokens, audio_tokens, after_tokens]
            combined_ids = torch.cat([
                tokens_before.input_ids,
                audio_token_ids,
                tokens_after.input_ids
            ], dim=1)
            
            # Mask
            combined_mask = torch.cat([
                tokens_before.attention_mask,
                torch.ones(audio_token_ids.shape, device=device),
                tokens_after.attention_mask
            ], dim=1)
            
            # Add Prefix (e.g., BOS) if provided
            if prefix_embeds is not None:
                # Convert prefix embeddings to token IDs (approximate)
                # Actually, if prefix_embeds is provided, we should use it as embeddings
                # But for token-based approach, we'll skip prefix for now
                pass
            
            input_ids_list.append(combined_ids)
            attn_mask_list.append(combined_mask)
        
        # Padding
        max_len = max([ids.shape[1] for ids in input_ids_list])
        
        padded_ids = []
        padded_masks = []
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        for i in range(batch_size):
            curr_len = input_ids_list[i].shape[1]
            pad_len = max_len - curr_len
            
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), pad_token_id, dtype=torch.long, device=device)
                pad_msk = torch.zeros(1, pad_len, device=device)
                
                if padding_side == "right":
                    p_ids = torch.cat([input_ids_list[i], pad_ids], dim=1)
                    p_mask = torch.cat([attn_mask_list[i], pad_msk], dim=1)
                else: # left
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
    
    def _prompt_wrap_embeds(self, speech_embeds, speech_lens, prompt, multi_prompt=False, padding_side="right", prefix_embeds=None):
        """
        Wraps continuous speech embeddings with text prompts (original implementation).
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
            
            # Try to apply chat template if available
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                try:
                    # Use a unique placeholder to avoid tokenization issues
                    placeholder = "___SPEECH_PLACEHOLDER___"
                    
                    # Clean up legacy "User: ... Answer:" if present to avoid double prompting
                    clean_prompt = prompt.replace("User:", "").replace("Answer:", "").strip()
                    clean_prompt = clean_prompt.replace("<SpeechHere>", placeholder)
                    
                    # Prepare messages
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."}, 
                        {"role": "user", "content": clean_prompt}
                    ]
                    
                    # apply_chat_template returns a string
                    full_formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    if placeholder in full_formatted_prompt:
                        p_before, p_after = full_formatted_prompt.split(placeholder)
                    else:
                        # Fallback if placeholder was mangled
                        if "<SpeechHere>" not in prompt:
                            p_before, p_after = prompt, ""
                        else:
                            p_before, p_after = prompt.split("<SpeechHere>")
                except Exception as e:
                    # Fallback to raw string splitting if chat template fails
                    if "<SpeechHere>" not in prompt:
                        p_before, p_after = prompt, ""
                    else:
                        p_before, p_after = prompt.split("<SpeechHere>")
            else:
                # No chat template available, use simple splitting
                if "<SpeechHere>" not in prompt:
                    p_before, p_after = prompt, ""
                else:
                    p_before, p_after = prompt.split("<SpeechHere>")
            
            tokens_before = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            tokens_after = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)
            
            embeds_before = self._get_text_embeds(tokens_before.input_ids)
            embeds_after = self._get_text_embeds(tokens_after.input_ids)
            
            # Slice speech embeds to valid length using speech_lens
            valid_len = speech_lens[i]
            
            cur_speech_embeds = speech_embeds[i:i+1, :valid_len, :] # [1, valid_len, H]

            combined_embeds = torch.cat([embeds_before, cur_speech_embeds, embeds_after], dim=1)

            # Mask
            combined_mask = torch.cat([
                tokens_before.attention_mask, 
                torch.ones(cur_speech_embeds.shape[:2], device=device), 
                tokens_after.attention_mask
            ], dim=1)
            
            # Add Prefix (e.g., BOS) if provided
            if prefix_embeds is not None:
                # prefix_embeds is [B, L_prefix, H]
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
                else: # left
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
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        
        # 1. Encode Audio
        speech_embeds, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)
        
        # 2. Get Prompts
        prompts = samples.get("prompt", None)
        if prompts is None:
             raise ValueError("Prompt must be provided in samples for forward pass")

        # 3. Wrap Inputs
        multi_prompt = isinstance(prompts, list)
        inputs_embeds, attention_mask = self.prompt_wrap(speech_embeds, speech_lens, prompts, multi_prompt=multi_prompt, padding_side="right")
        
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
            
            curr_input_embeds = inputs_embeds[i:i+1] # [1, Max_Len, H] - keep batch dimension
            curr_mask = attention_mask[i:i+1] # [1, Max_Len] - keep batch dimension
            
            # Get valid length (number of non-padding tokens)
            valid_len = curr_mask.sum(dim=1).int().item()  # Sum over sequence dimension
            
            # Slice to valid length (remove right padding)
            real_input_embeds = curr_input_embeds[:, :valid_len]  # [1, valid_len, H]
            real_mask = curr_mask[:, :valid_len]  # [1, valid_len] - should be all 1s
            
            # Concat: [BOS, Prompt+Speech, Target]
            # Since Prompt+Speech already includes the start of the assistant turn (via Chat Template),
            # we just append the Target which is the assistant's answer content.
            # And usually the chat template ends with <|im_start|>assistant\n
            
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
                # Right Padding for Training (Forward)
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

    def get_last_token_logits(self, samples, prompts=None):
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        
        # Encode
        speech_embeds, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)
        
        # Prepare Prompts
        if prompts is None:
            if "prompt" in samples:
                prompts = samples["prompt"]
            else:
                 raise ValueError("Prompts must be provided")

        multi_prompt = isinstance(prompts, list)

        # Prepare BOS
        batch_size = speech_embeds.shape[0]
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        
        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.long,
            device=speech_embeds.device,
        ) * bos_token_id
        bos_embeds = self._get_text_embeds(bos)

        # Wrap with Left Padding and Prefix (BOS)
        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_embeds, 
            speech_lens, 
            prompts, 
            multi_prompt=multi_prompt, 
            padding_side="left",
            prefix_embeds=bos_embeds
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
        raw_wav = samples["raw_wav"]
        audio_attention_mask = samples.get("audio_attention_mask", None)
        
        # Encode
        speech_embeds, speech_lens = self.encode_speech(raw_wav, audio_attention_mask)
        
        # Prepare Prompts
        if prompts is None:
            if "prompt" in samples:
                prompts = samples["prompt"]
            else:
                 raise ValueError("Prompts must be provided")

        multi_prompt = isinstance(prompts, list)

        # Prepare BOS
        batch_size = speech_embeds.shape[0]
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        
        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.long,
            device=speech_embeds.device,
        ) * bos_token_id
        bos_embeds = self._get_text_embeds(bos)

        # Wrap with Left Padding and Prefix (BOS)
        inputs_embeds, attention_mask = self.prompt_wrap(
            speech_embeds, 
            speech_lens, 
            prompts, 
            multi_prompt=multi_prompt, 
            padding_side="left",
            prefix_embeds=bos_embeds
        )
        
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
