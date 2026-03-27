import os
import argparse
import logging
import torch
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from contextlib import nullcontext
import random
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_clean.audiollm_base_opensmile import AudioDeepfakeLLM
from dataset import AudioDeepfakeDataset
from huggingface_hub import login

# Initialize Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# login(token="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")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model_generative(checkpoint_dir, epoch, device, compile_model=False):
    """
    Loads the model for text generation using the exact robust loading logic from your original script.
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in {checkpoint_dir}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loading configuration from {config_path}")
    model_cfg = config["model"]
    
    # 1. Initialize Base Model Architecture
    model = AudioDeepfakeLLM(
        llm_path=model_cfg["llm_path"],
        llm_type=model_cfg.get("llm_type", "qwen2"),
        audio_encoder_path=model_cfg.get("audio_encoder_path", None),
        audio_encoder_type=model_cfg.get("audio_encoder_type", "whisper"),
        audio_encoder_source=model_cfg.get("audio_encoder_source", "huggingface"),
        freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True),
        # Force lora=True in init if config has it, but we handle loading manually
        lora=model_cfg.get("lora", True), 
        lora_rank=model_cfg.get("lora_rank", 64),
        lora_alpha=model_cfg.get("lora_alpha", 16),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        special_tokens=model_cfg.get("special_tokens", []),
        device_8bit=0,
        # Audio Encoder Params
        speechtokenizer_num_layers=model_cfg.get("speechtokenizer_num_layers", 8),
        speechtokenizer_codebook_size=model_cfg.get("speechtokenizer_codebook_size", 1024),
        speechtokenizer_return_discrete=model_cfg.get("speechtokenizer_return_discrete", True),
        dac_num_codebooks=model_cfg.get("dac_num_codebooks", 12),
        dac_codebook_size=model_cfg.get("dac_codebook_size", 1024),
        dac_model_type=model_cfg.get("dac_model_type", "16khz"),
        dac_return_discrete=model_cfg.get("dac_return_discrete", True),
        sqcodec_num_quantizers=model_cfg.get("sqcodec_num_quantizers", 4),
        sqcodec_num_levels=model_cfg.get("sqcodec_num_levels", 4096),
        use_weighted_layer_sum=model_cfg.get("use_weighted_layer_sum", False),
        weighted_layer_norm=model_cfg.get("weighted_layer_norm", "none"),
        use_opensmile=model_cfg.get("use_opensmile", False),
        opensmile_only=model_cfg.get("opensmile_only", False),
    )

    # 2. Determine File Paths
    if epoch == "best":
        lora_dir_name = "lora_best"
        ckpt_filename = "checkpoint_best.pt"
        proj_filename = "proj_best.pt"
        enc_filename = "audio_encoder_best.pt"
    else:
        lora_dir_name = f"lora_epoch_{epoch}"
        ckpt_filename = f"checkpoint_epoch_{epoch}.pt"
        proj_filename = f"proj_epoch_{epoch}.pt"
        enc_filename = f"audio_encoder_epoch_{epoch}.pt"

    lora_path = os.path.join(checkpoint_dir, lora_dir_name)
    ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
    
    # 3. Check for LoRA Weights FIRST
    if os.path.exists(lora_path):
        logger.info(f"✅ Found LoRA weights at {lora_path}")
        
        # Load auxiliary components first
        proj_path = os.path.join(checkpoint_dir, proj_filename)
        if model.speech_proj is not None and os.path.exists(proj_path):
            logger.info(f"Loading Projector from {proj_path}")
            model.speech_proj.load_state_dict(torch.load(proj_path, map_location="cpu", weights_only=True))
            
        enc_path = os.path.join(checkpoint_dir, enc_filename)
        if os.path.exists(enc_path):
            logger.info(f"Loading Audio Encoder from {enc_path}")
            model.audio_encoder.load_state_dict(torch.load(enc_path, map_location="cpu", weights_only=True))

        # Load LoRA Adapters
        logger.info("Loading LoRA adapters...")
        try:
            model.llm_model.load_adapter(lora_path, adapter_name="default")
        except Exception as e:
            logger.error(f"Failed to load LoRA adapters: {e}")
            raise e

    # 4. Fallback to Full Checkpoint (Using YOUR exact manual loading logic)
    elif os.path.exists(ckpt_path):
        logger.info(f"⚠️ LoRA path not found. Loading Full Checkpoint from {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Manual loading with key renaming to avoid shape mismatch
        model_state_dict = model.state_dict()
        pretrained_dict = {}
        shape_mismatch_keys = []
        
        for k, v in checkpoint.items():
            candidates = [k]
            if "audio_encoder.encoder.encoder." in k:
                candidates.append(k.replace("audio_encoder.encoder.encoder.", "audio_encoder.encoder."))
            if "audio_encoder.encoder." in k and "audio_encoder.encoder.encoder." not in k:
                candidates.append(k.replace("audio_encoder.encoder.", "audio_encoder.encoder.encoder."))
            if "audio_encoder.layers." in k:
                candidates.append(k.replace("audio_encoder.layers.", "audio_encoder.encoder.layers."))
            if "audio_encoder.encoder.layers." in k:
                candidates.append(k.replace("audio_encoder.encoder.layers.", "audio_encoder.layers."))
            if "llm_model.base_model.model.model." in k:
                candidates.append(k.replace("llm_model.base_model.model.model.", "llm_model.model."))
            
            match_found = False
            for cand in candidates:
                if cand in model_state_dict:
                    # STRICT SHAPE CHECK: Only add if shapes match
                    if v.shape == model_state_dict[cand].shape:
                        pretrained_dict[cand] = v
                        match_found = True
                        break
                    else:
                        shape_mismatch_keys.append(cand)
                        logger.warning(
                            f"Shape mismatch for {cand} (from {k}): checkpoint {v.shape} vs model {model_state_dict[cand].shape}"
                        )
            
            if not match_found and "audio_encoder" in k:
                pass

        # Load the constructed dictionary
        model.load_state_dict(pretrained_dict, strict=False)
        
        logger.info(f"Checkpoint loaded (partial). Shape mismatches ignored: {len(shape_mismatch_keys)}")
        if shape_mismatch_keys:
            logger.info(f"First 5 mismatched keys: {shape_mismatch_keys[:5]}")
            
        if any("speech_proj" in k for k in pretrained_dict.keys()):
            logger.info("Confirmed: speech_proj weights loaded from checkpoint.")
        else:
            logger.warning("WARNING: speech_proj weights NOT loaded from checkpoint!")
            
        if any("audio_encoder" in k for k in pretrained_dict.keys()):
            logger.info("Confirmed: audio_encoder weights loaded from checkpoint.")
        else:
            logger.warning("WARNING: audio_encoder weights NOT loaded from checkpoint!")

    else:
        raise FileNotFoundError(
            f"Could not find weights! Checked:\n - LoRA: {lora_path}\n - Checkpoint: {ckpt_path}"
        )

    # 5. Move to Device
    model.to(device)
    model.eval()
    
    if compile_model:
        logger.info("Compiling model...")
        model = torch.compile(model)
        
    return model

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = load_model_generative(
        args.checkpoint_dir, 
        args.epoch, 
        device, 
        compile_model=args.compile
    )
    
    # 2. Load Dataset
    logger.info(f"Loading dataset from {args.input_jsonl}")
    dataset = AudioDeepfakeDataset(args.input_jsonl, prompt_path=None)
    
    # Override prompt if provided
    if args.prompt:
        dataset.prompts = [args.prompt]
        logger.info(f"Overriding prompt with: {args.prompt}")

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers
    )

    # 3. Generation Loop
    results = []
    logger.info(f"Starting generation on {len(dataset)} samples...")
    
    # Mixed Precision Context
    amp_ctx = nullcontext()
    if args.mixed_precision == 'fp16':
        amp_ctx = torch.amp.autocast('cuda', dtype=torch.float16)
    elif args.mixed_precision == 'bf16':
        amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            # Move data to device
            raw_wav = batch["raw_wav"].to(device)
            audio_attention_mask = batch.get("audio_attention_mask")
            if audio_attention_mask is not None:
                audio_attention_mask = audio_attention_mask.to(device)
            
            prompts = batch["prompt"]
            texts = batch["text"]
            paths = batch["path"]
            
            samples = {
                "raw_wav": raw_wav,
                "audio_attention_mask": audio_attention_mask,
                "prompt": prompts
            }
            
            # Generate
            with amp_ctx:
                generated_texts = model.generate(
                    samples,
                    generate_cfg={
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": args.do_sample,
                        "temperature": args.temperature,
                        "top_p": args.top_p
                    },
                    prompts=prompts
                )
            
            # Collect results
            for i in range(len(paths)):
                full_output = generated_texts[i]
                print(full_output)
                results.append({
                    "path": paths[i],
                    "ground_truth": texts[i],
                    "prompt": prompts[i],
                    "generated_text": full_output
                })

    # 4. Save Output
    output_file = args.output_file
    if not output_file:
        input_name = os.path.splitext(os.path.basename(args.input_jsonl))[0]
        output_file = os.path.join(args.checkpoint_dir, f"generated_{input_name}_{args.epoch}.csv")
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logger.info(f"Generation complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path containing config.json and weights")
    parser.add_argument("--epoch", type=str, default="best", help="Epoch number or 'best'")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save output CSV")
    
    # Generation Config
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (vs greedy search)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=[None, 'fp16', 'bf16'])
    parser.add_argument("--compile", action="store_true")
    
    # Prompt Override
    parser.add_argument("--prompt", type=str, default=None, help="Override prompt from dataset")
    
    args = parser.parse_args()
    main(args)