"""Train AudioDeepfakeLLM with Hugging Face TRL GRPO."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import Dataset as HFDataset
from trl import GRPOConfig
from transformers.trainer_utils import get_last_checkpoint

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
from train_grpo import _load_pretrained
from trl_grpo import make_audio_grpo_trainer


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
