"""Train AudioDeepfakeLLM with Hugging Face TRL GRPO."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import Dataset as HFDataset
from trl import GRPOConfig
from transformers.trainer_utils import get_last_checkpoint
import torch

from dataset import AudioDeepfakeDataset
from grpo_utils import compute_reward
from train_common import (
    build_model,
    distributed_barrier,
    init_distributed,
    load_config,
    maybe_hf_login,
    save_config,
    set_seed,
    setup_logging,
)
from trl_grpo import make_audio_grpo_trainer

def _load_pretrained(
    model,
    checkpoint_dir: str,
    checkpoint_epoch: str,
) -> None:
    ckpt = Path(checkpoint_dir) / f"checkpoint_{checkpoint_epoch}.pt"
    if ckpt.is_file():
        logging.info("Loading full checkpoint %s", ckpt)
        state = torch.load(ckpt, map_location="cpu")
        model_state = model.state_dict()
        loaded = {}
        for key, value in state.items():
            if key in model_state and value.shape == model_state[key].shape:
                loaded[key] = value
        model.load_state_dict(loaded, strict=False)
        logging.info("Loaded %s matching tensors from checkpoint", len(loaded))
        return

    proj = Path(checkpoint_dir) / f"proj_{checkpoint_epoch}.pt"
    enc = Path(checkpoint_dir) / f"audio_encoder_{checkpoint_epoch}.pt"
    mlp = Path(checkpoint_dir) / f"mlp_adaptor_{checkpoint_epoch}.pt"
    if model.speech_proj is not None and proj.is_file():
        logging.info("Loading projector from %s", proj)
        model.speech_proj.load_state_dict(torch.load(proj, map_location="cpu"))
    if enc.is_file():
        logging.info("Loading audio encoder from %s", enc)
        model.audio_encoder.load_state_dict(torch.load(enc, map_location="cpu"))
    if model.use_mlp_adaptor and model.mlp_adaptor is not None and mlp.is_file():
        logging.info("Loading MLP adaptor from %s", mlp)
        model.mlp_adaptor.load_state_dict(torch.load(mlp, map_location="cpu"))
    if not proj.is_file() and not enc.is_file():
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
    lora_dir = Path(checkpoint_dir) / (
        f"lora_{checkpoint_epoch}"
        if checkpoint_epoch != "best"
        else "lora_best"
    )

    if lora_dir.is_dir():
        logging.info("Loading LoRA adapter from %s", lora_dir)
        model.llm_model.load_adapter(
            str(lora_dir),
            adapter_name="default",
            is_trainable=True,
        )




def reward_func(prompts, completions, text, **kwargs):
    return [compute_reward(completion, target) for completion, target in zip(completions, text)]


def _dataset(config: dict, split: str) -> HFDataset:
    path = config["data"][f"{split}_path"]
    source = AudioDeepfakeDataset(path, prompt_path=config["data"].get("prompt_path"))
    rows = []
    for idx, item in enumerate(source.data):
        rows.append({
            "path": item["path"],
            "text": item.get("text", "Real"),
            "prompt": source.prompts[idx % len(source.prompts)],
        })
    return HFDataset.from_list(rows)


def main(
    config_path: str,
    pretrained_checkpoint: str,
    checkpoint_epoch: str,
    hf_token: str | None,
    resume_from_checkpoint: str | None,
):
    device, is_main_process = init_distributed()
    maybe_hf_login(hf_token)
    config = load_config(config_path)
    output_dir = config.get("output_dir", "./output_trl_grpo")
    setup_logging(output_dir)
    if is_main_process:
        save_config(config, output_dir)
    distributed_barrier()
    set_seed(config["train"].get("seed", 42))

    model = build_model(config["model"]).to(device)
    _load_pretrained(model, pretrained_checkpoint, checkpoint_epoch)

    grpo_cfg = config["train"].get("grpo", {})
    train_cfg = config["train"]
    args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg.get("grad_accum_steps", 1),
        learning_rate=float(train_cfg.get("lr", 1e-5)),
        num_train_epochs=train_cfg["epochs"],
        num_generations=int(grpo_cfg.get("num_rollouts", 4)),
        max_completion_length=int(grpo_cfg.get("max_new_tokens", 128)),
        temperature=float(grpo_cfg.get("temperature", 0.8)),
        top_p=float(grpo_cfg.get("top_p", 0.9)),
        repetition_penalty=float(grpo_cfg.get("repetition_penalty", 1.1)),
        epsilon=float(grpo_cfg.get("clip_epsilon", grpo_cfg.get("clip_ratio", 0.2))),
        beta=float(grpo_cfg.get("kl_beta", 0.0)),
        remove_unused_columns=False,
        use_vllm=False,
        gradient_checkpointing=False,
        logging_steps=grpo_cfg.get("log_freq", 50),
        save_steps=grpo_cfg.get("save_every_steps", 200),
        report_to=[],
        bf16=bool(train_cfg.get("bf16", False)),
        fp16=bool(train_cfg.get("fp16", False)),
    )
    args.save_safetensors = False

    trainer = make_audio_grpo_trainer(
        model,
        args=args,
        reward_funcs=reward_func,
        train_dataset=_dataset(config, "train"),
        eval_dataset=_dataset(config, "val") if config["data"].get("val_path") else None,
        processing_class=model.tokenizer,
    )
    resume_checkpoint = resume_from_checkpoint or get_last_checkpoint(output_dir)
    if resume_checkpoint:
        logging.info("Resuming TRL training from %s", resume_checkpoint)
    else:
        logging.info("No TRL checkpoint found under %s; starting fresh", output_dir)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        # Keep the repository's standard evaluation format alongside TRL files.
        from train_common import save_best_checkpoint

        save_best_checkpoint(
            model,
            output_dir,
            config,
            epoch=int(train_cfg["epochs"]),
            val_loss=float("nan"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--checkpoint_epoch", default="best")
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="TRL checkpoint directory; defaults to the latest checkpoint-* in output_dir",
    )
    parser.add_argument("--hf_token", default=None)
    arguments = parser.parse_args()
    main(
        arguments.config,
        arguments.pretrained_checkpoint,
        arguments.checkpoint_epoch,
        arguments.hf_token,
        arguments.resume_from_checkpoint,
    )
