"""TRL GRPO adapter for the audio-conditioned model.

TRL owns rollout grouping, advantages, clipping, KL, optimization, and
checkpointing. This module only bridges TRL's text-oriented hooks to the
repository's audio-aware generation and completion-logit methods.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from dataset import load_audio_file
from grpo import get_log_probs


class AudioDeepfakeTRLModel(nn.Module):
    """Expose the metadata and hooks expected by ``GRPOTrainer``."""
    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None  # trainer/transformers may probe this too on some versions
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, audio_model):
        super().__init__()
        self.audio_model = audio_model
        # Keep the compatibility alias out of nn.Module's registered children.
        object.__setattr__(self, "llm_model", getattr(audio_model, "llm_model", audio_model))
        self.config = getattr(
            self.llm_model,
            "config",
            SimpleNamespace(_name_or_path="audio-deepfake-llm"),
        )
        self.warnings_issued = {}
        self.name_or_path = getattr(self.config, "_name_or_path", "audio-deepfake-llm")
        self._generation_config = getattr(self.llm_model, "generation_config", None)
        if self._generation_config is None:
            try:
                from transformers import GenerationConfig

                self._generation_config = GenerationConfig.from_model_config(self.config)
            except Exception:
                self._generation_config = SimpleNamespace(
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )

    @property
    def generation_config(self):
        if self._generation_config is None:
            self._generation_config = getattr(self.llm_model, "generation_config", None)
        if self._generation_config is None:
            try:
                from transformers import GenerationConfig

                self._generation_config = GenerationConfig.from_model_config(self.config)
            except Exception:
                self._generation_config = SimpleNamespace(
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value):
        self._generation_config = value

    @property
    def tokenizer(self):
        return self.audio_model.tokenizer

    def add_model_tags(self, tags):
        if hasattr(self.audio_model.llm_model, "add_model_tags"):
            self.audio_model.llm_model.add_model_tags(tags)

    def forward(self, raw_wav=None, audio_attention_mask=None, prompt=None, **kwargs):
        samples = {
            "raw_wav": raw_wav,
            "audio_attention_mask": audio_attention_mask,
            "prompt": prompt,
        }
        return self.audio_model(samples)

    def generate(self, samples, generate_cfg, prompts=None, return_outputs=False):
        return self.audio_model.generate(
            samples, generate_cfg, prompts=prompts, return_outputs=return_outputs
        )

    def compute_logits_for_completions(self, samples, completion_ids):
        return self.audio_model.compute_logits_for_completions(samples, completion_ids)


class AudioGRPOTrainer(__import__("trl", fromlist=["GRPOTrainer"]).GRPOTrainer):
    """GRPOTrainer with audio-aware rollout and policy log-prob computation."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("args")
        self.audio_beta = float(getattr(config, "beta", 0.0))
        if config is not None:
            config.beta = 0.0
        super().__init__(*args, **kwargs)
        tokenizer = self.model.tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self._reference_state_dict = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.audio_model.named_parameters()
        }

    def _audio_batch(self, inputs):
        device = self.accelerator.device
        raw_wavs = [load_audio_file(item["path"]) for item in inputs]
        raw_wav = torch.nn.utils.rnn.pad_sequence(
            raw_wavs, batch_first=True
        ).to(device)
        lengths = torch.tensor(
            [raw_wav.numel() for raw_wav in raw_wavs], device=device
        )
        audio_attention_mask = (
            torch.arange(raw_wav.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
        )
        return {
            "raw_wav": raw_wav,
            "audio_attention_mask": audio_attention_mask,
            "prompt": [item["prompt"] for item in inputs],
        }

    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        samples = self._audio_batch(inputs)
        generation_cfg = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }

        with torch.inference_mode():
            generated_texts, completion_ids, _ = self.model.generate(
                samples,
                generation_cfg,
                prompts=samples["prompt"],
                return_outputs=True,
            )

        completion_mask = completion_ids.ne(self.pad_token_id).long()
        if self.eos_token_id is not None:
            eos = completion_ids.eq(self.eos_token_id)
            after_eos = eos.cumsum(dim=1) - eos.long() > 0
            completion_mask = completion_mask * (~after_eos).long()

        prompts = samples["prompt"]
        completion_ids_list = [
            row[mask.bool()].tolist()
            for row, mask in zip(completion_ids, completion_mask)
        ]
        rewards_per_func = self._calculate_rewards(
            inputs, prompts, generated_texts, completion_ids_list
        )
        rewards = (rewards_per_func * self.reward_weights.to(device)).nansum(dim=1)
        mean_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        advantages = rewards - mean_rewards.repeat_interleave(self.num_generations)
        std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        std_rewards = std_rewards.repeat_interleave(self.num_generations)
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)
        local_size = len(prompts)
        process_slice = slice(
            self.accelerator.process_index * local_size,
            (self.accelerator.process_index + 1) * local_size,
        )
        advantages = advantages[process_slice]
        rewards = rewards[process_slice]

        prompt_tokens = self.processing_class(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        prompt_ids = prompt_tokens["input_ids"].to(device)
        prompt_mask = prompt_tokens["attention_mask"].to(device)
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "raw_wav": samples["raw_wav"],
            "audio_attention_mask": samples["audio_attention_mask"],
            "prompt": prompts,
            "text": [item["text"] for item in inputs],
            "advantages": advantages,
            "num_items_in_batch": completion_mask.sum(),
        }
        return output

    def _compute_loss(self, model, inputs):
        model = self.accelerator.unwrap_model(model)
        samples = {
            "raw_wav": inputs["raw_wav"],
            "audio_attention_mask": inputs["audio_attention_mask"],
            "prompt": inputs["prompt"],
        }
        logits = model.compute_logits_for_completions(samples, inputs["completion_ids"])
        log_probs = get_log_probs(logits, inputs["completion_ids"])
        completion_mask = inputs["completion_mask"].float()
        advantages = inputs["advantages"]

        old_log_probs = inputs.get("old_per_token_logps")
        if old_log_probs is None:
            old_log_probs = log_probs.detach()
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio, 1 - self.epsilon_low, 1 + self.epsilon_high
        )
        per_token_loss = -torch.minimum(
            ratio * advantages.unsqueeze(1),
            clipped_ratio * advantages.unsqueeze(1),
        )

        if self.audio_beta:
            reference_model = model.audio_model
            current_state = {}
            for name, parameter in reference_model.named_parameters():
                if name in self._reference_state_dict:
                    current_state[name] = parameter.detach().clone()
                    parameter.data.copy_(self._reference_state_dict[name])
            try:
                with torch.no_grad():
                    reference_logits = model.compute_logits_for_completions(
                        samples, inputs["completion_ids"]
                    )
                    reference_log_probs = get_log_probs(
                        reference_logits, inputs["completion_ids"]
                    )
            finally:
                for name, parameter in reference_model.named_parameters():
                    if name in current_state:
                        parameter.data.copy_(current_state[name])
            per_token_loss = per_token_loss + self.audio_beta * (
                torch.exp(reference_log_probs - log_probs)
                - (reference_log_probs - log_probs)
                - 1
            )

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp_min(1.0)
        return loss / self.current_gradient_accumulation_steps


def make_audio_grpo_trainer(audio_model, **kwargs):
    """Build the TRL trainer and reject unsupported vLLM settings early."""
    args = kwargs["args"]
    if args.use_vllm:
        raise ValueError("AudioDeepfakeLLM requires use_vllm=False in TRL GRPOConfig")
    return AudioGRPOTrainer(AudioDeepfakeTRLModel(audio_model), **kwargs)