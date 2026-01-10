import os
import argparse
import yaml
import torch
import librosa
import numpy as np
from peft import PeftModel

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.audio_llm import AudioDeepfakeLLM
from huggingface_hub import login
login(token="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")

def load_model(config_path, checkpoint_dir, epoch, device):
    # Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize Model
    model_cfg = config["model"]
    model = AudioDeepfakeLLM(
        llm_path=model_cfg["llm_path"],
        llm_type=model_cfg.get("llm_type", "qwen2"),
        audio_encoder_path=model_cfg["audio_encoder_path"],
        audio_encoder_type=model_cfg.get("audio_encoder_type", "whisper"),
        audio_encoder_source=model_cfg.get("audio_encoder_source", "huggingface"),
        freeze_audio_encoder=model_cfg.get("freeze_audio_encoder", True),
        lora=model_cfg.get("lora", True),
        lora_rank=model_cfg.get("lora_rank", 64),
        lora_alpha=model_cfg.get("lora_alpha", 16),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        special_tokens=model_cfg.get("special_tokens", []),
        device_8bit=0 
    )
    
    # Load Projector Weights
    proj_path = os.path.join(checkpoint_dir, f"proj_epoch_{epoch}.pt")
    if os.path.exists(proj_path):
        print(f"Loading projector weights from {proj_path}")
        model.speech_proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
    else:
        print(f"Warning: Projector weights not found at {proj_path}")

    # Load Audio Encoder Weights if they exist
    audio_enc_path = os.path.join(checkpoint_dir, f"audio_encoder_epoch_{epoch}.pt")
    if os.path.exists(audio_enc_path):
        print(f"Loading audio encoder weights from {audio_enc_path}")
        model.audio_encoder.load_state_dict(torch.load(audio_enc_path, map_location="cpu"))
    
    # Load LoRA Weights
    lora_path = os.path.join(checkpoint_dir, f"lora_epoch_{epoch}")
    if os.path.exists(lora_path) and config["model"].get("lora", True):
        print(f"Loading LoRA weights from {lora_path}")
        # The model initialized above creates a fresh LoRA adapter. 
        # We need to load the trained weights into it.
        # Since get_peft_model was called in __init__, model.llm_model is a PeftModel.
        # We can use load_adapter (if we want to add) or just load the state dict if it matches.
        # But standard way is PeftModel.from_pretrained.
        
        # However, model.llm_model is already a PeftModel wrapping the base model.
        # We can try to load the adapter.
        model.llm_model.load_adapter(lora_path, adapter_name="default")
    else:
        print(f"Warning: LoRA weights not found at {lora_path} or LoRA disabled.")

    model.to(device)
    model.eval()
    return model

def process_audio(audio_path, sample_rate=16000):
    try:
        audio, _ = librosa.load(audio_path, sr=sample_rate)
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return None
    
    # Convert to Tensor [1, T]
    raw_wav = torch.from_numpy(audio).float().unsqueeze(0)
    
    # Attention mask (all valid)
    audio_attention_mask = torch.ones(raw_wav.shape[0], raw_wav.shape[1]).long()
    
    # Padding mask (False for valid) - mostly for Fairseq/WavLM
    padding_mask = torch.zeros(raw_wav.shape[0], raw_wav.shape[1]).bool()
    
    return raw_wav, audio_attention_mask, padding_mask

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = load_model(args.config, args.checkpoint_dir, args.epoch, device)
    
    # Prepare Input
    raw_wav, att_mask, pad_mask = process_audio(args.audio_path)
    if raw_wav is None:
        return

    raw_wav = raw_wav.to(device)
    att_mask = att_mask.to(device)
    pad_mask = pad_mask.to(device)
    
    samples = {
        "raw_wav": raw_wav,
        "audio_attention_mask": att_mask,
        "padding_mask": pad_mask,
        "prompt": [args.prompt]
    }
    
    # Generate
    with torch.no_grad():
        output_text = model.generate(samples, generate_cfg={"max_new_tokens": 200, "do_sample": False})
    
    print("-" * 30)
    print(f"Input Audio: {args.audio_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {output_text[0]}")
    
    # Basic Hallucination Check
    pred_lower = output_text[0].lower()
    is_valid = ("real" in pred_lower and "fake" not in pred_lower) or \
               ("fake" in pred_lower and "real" not in pred_lower)
    
    if not is_valid:
        print("WARNING: Possible Hallucination or Ambiguous Output (Did not strictly output 'Real' or 'Fake')")
    
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoints (output_dir)")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to load")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file")
    parser.add_argument("--prompt", type=str, default="User: <SpeechHere> Is this audio real or fake? Answer: ", help="Text prompt")
    
    args = parser.parse_args()
    main(args)
