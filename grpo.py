from __future__ import annotations

import logging
import time

import torch
from torch.utils.data import DataLoader

from grpo_utils import compute_reward
from train_common import batch_to_samples
from train_common import average_gradients, distributed_barrier


def _default_generate_cfg(grpo_cfg: dict) -> dict:
    return {
        "max_new_tokens": grpo_cfg.get("max_new_tokens", 128),
        "do_sample": grpo_cfg.get("do_sample", True),
        "temperature": grpo_cfg.get("temperature", 0.8),
        "top_p": grpo_cfg.get("top_p", 0.9),
        "repetition_penalty": grpo_cfg.get("repetition_penalty", 1.1),
    }


def _repeat_samples(samples: dict, num_rollouts: int) -> dict:
    """Expand batch dim B -> B*num_rollouts so all rollouts run as one batched call."""
    out = {}
    for key, value in samples.items():
        if torch.is_tensor(value):
            out[key] = value.repeat_interleave(num_rollouts, dim=0)
        else:  # lists: prompt, text
            out[key] = [item for item in value for _ in range(num_rollouts)]
    return out


def _process_completions(all_texts, completion_ids, model):
    pad_token_id = model.tokenizer.pad_token_id
    eos_token_id = model.tokenizer.eos_token_id

    mask = (completion_ids != pad_token_id).float()
    if eos_token_id is not None:
        eos = completion_ids.eq(eos_token_id)
        after_first_eos = (eos.cumsum(dim=1) - eos.float()) > 0
        mask = mask * (~after_first_eos).float()

    cleaned_texts = []
    for text in all_texts:
        if "<|endoftext|>" in text:
            text = text.split("<|endoftext|>")[0] + "<|endoftext|>"
        cleaned_texts.append(text)

    return completion_ids, mask, cleaned_texts


def get_log_probs(logits, labels):
    """
    Compute log softmax for selected tokens only.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len) - token IDs

    Returns:
        log_probs: (batch_size, seq_len) - log probabilities of selected tokens
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    batch_size, seq_len, vocab_size = logits.shape

    log_probs_flat = log_probs.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    selected_log_probs = log_probs_flat.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
    selected_log_probs = selected_log_probs.view(batch_size, seq_len)

    return selected_log_probs


def get_reference_log_probs(policy_model, initial_state_dict, samples, completion_ids):
    """
    Compute log-probabilities under the frozen *initial* policy without holding a
    second full copy of the model in memory.

    Instead of a standalone ref_model (build_ref_model's deepcopy, ~2x model
    memory on the GPU), this temporarily overwrites the live policy model's own
    parameters with the initial snapshot, runs the forward pass, then restores
    the current (training) weights. Net GPU cost is ~1x model instead of ~2x,
    at the price of two extra parameter copies per call.

    initial_state_dict is expected to live on CPU (see run_training_loop_grpo);
    `.copy_()` handles the CPU->GPU / GPU->CPU transfer transparently.
    """
    current_state = {}
    for name, param in policy_model.named_parameters():
        if name in initial_state_dict:
            current_state[name] = param.data.clone()
            param.data.copy_(initial_state_dict[name])

    try:
        ref_logits = policy_model.compute_logits_for_completions(samples, completion_ids)
        ref_logprobs = get_log_probs(ref_logits, completion_ids)
    finally:
        # Always restore, even if the forward pass above raises.
        for name, param in policy_model.named_parameters():
            if name in current_state:
                param.data.copy_(current_state[name])

    return ref_logprobs


def grpo_train_step(policy_model, initial_state_dict, samples, grpo_cfg, *, step=0):
    generate_cfg = _default_generate_cfg(grpo_cfg)
    num_rollouts = int(grpo_cfg.get("num_rollouts", 4))
    beta = grpo_cfg.get("kl_beta", 0.0)
    # clip_ratio = grpo_cfg.get("clip_ratio", 0.2)
    log_freq = grpo_cfg.get("log_freq", 50)

    batch_size = len(samples["prompt"])
    device = samples["raw_wav"].device
    expanded = _repeat_samples(samples, num_rollouts)

    with torch.no_grad():
        generated_texts, completion_ids, _ = policy_model.generate(
            expanded, generate_cfg, prompts=expanded["prompt"], return_outputs=True
        )
        completion_ids, completion_mask, cleaned_texts = _process_completions(
            generated_texts, completion_ids, policy_model
        )
        rewards = torch.tensor(
            [compute_reward(p, g) for p, g in zip(cleaned_texts, expanded["text"])],
            device=device, dtype=torch.float32,
        )

        # Reference log-probs via weight swap instead of a second model instance.
        ref_logprobs = get_reference_log_probs(
            policy_model, initial_state_dict, expanded, completion_ids
        )

        # Advantages
        reward_matrix = rewards.view(batch_size, num_rollouts)
        reward_mean = reward_matrix.mean(dim=1, keepdim=True)
        reward_std = reward_matrix.std(dim=1, keepdim=True).clamp_min(1e-6)
        advantages = ((reward_matrix - reward_mean) / reward_std).reshape(-1)
        advantages = advantages.detach()

    policy_model.train()
    logits = policy_model.compute_logits_for_completions(expanded, completion_ids)
    log_probs = get_log_probs(logits, completion_ids)

    masked_log_probs = log_probs * completion_mask
    masked_ref_logprobs = ref_logprobs * completion_mask
    token_count = completion_mask.sum(dim=1).clamp_min(1.0)

    # Non-negative KL (Option A)
    log_ratio =  masked_ref_logprobs -masked_log_probs
    kl_per_token = torch.exp(log_ratio) - log_ratio - 1

    kl_term = (
        (kl_per_token * completion_mask).sum(dim=1) / token_count
    ).mean()

    ratio = torch.exp(masked_log_probs - masked_log_probs.detach())
    policy_loss = -(
        ratio * advantages.unsqueeze(1) * completion_mask
    ).sum(dim=1) / token_count
    policy_loss = policy_loss.mean()

    loss = policy_loss + beta * kl_term

    if step % log_freq == 0:
        logging.info(
            "[GRPO] step %d | loss=%.4f | kl=%.4f | reward_mean=%.4f | reward_std=%.4f",
            step, loss.item(), kl_term.item(),
            rewards.mean().item(), rewards.std().item(),
        )

    return loss


def train_one_epoch_grpo(
    policy_model,
    initial_state_dict,
    train_loader: DataLoader,
    optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    grad_accum_steps: int,
    grpo_cfg: dict,
    *,
    output_dir: str,
    config: dict,
    start_step: int = 0,
    global_step: int = 0,
    save_every: int = 200,
) -> tuple[float, int]:
    from train_common import save_step_checkpoint  # local import, avoid circularity

    policy_model.train()
    total_loss = 0.0
    steps_done = 0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)

    step = start_step  # in case the loader is empty / all steps skipped
    for step, batch in enumerate(train_loader):
        if step < start_step:
            continue  # fast-forward past already-completed steps this epoch

        samples = batch_to_samples(batch, device)

        loss = grpo_train_step(
            policy_model, initial_state_dict, samples, grpo_cfg, step=step
        ) / grad_accum_steps
        loss.backward()
        total_loss += loss.item() * grad_accum_steps
        steps_done += 1
        global_step += 1

        if (step + 1) % grad_accum_steps == 0:
            # Accumulate locally and synchronize once per optimizer update.
            average_gradients(policy_model)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % grpo_cfg.get("log_freq", 50) == 0:
            logging.info(
                "[GRPO] Epoch %s/%s | step %s/%s | loss %.4f",
                epoch + 1, num_epochs, step, len(train_loader), loss.item() * grad_accum_steps,
            )

        is_main_process = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        if is_main_process and save_every > 0 and (step + 1) % save_every == 0:
            save_step_checkpoint(
                policy_model, optimizer, output_dir, config,
                epoch=epoch, step=step + 1, global_step=global_step,
            )

    if steps_done > 0 and (step + 1) % grad_accum_steps != 0:
        average_gradients(policy_model)
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(steps_done, 1)
    logging.info("[GRPO] Epoch %s train loss %.4f (%.1fs)", epoch + 1, avg_loss, time.time() - start)
    return avg_loss, global_step


def run_training_loop_grpo(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    config: dict,
    output_dir: str,
    device: torch.device,
    *,
    num_epochs: int | None = None,
) -> None:
    from train_common import (
        load_step_checkpoint,
        save_best_checkpoint,
        train_one_epoch,
        validate,
    )

    is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    grpo_cfg = config["train"].get("grpo", {})
    grpo_start_epoch = grpo_cfg.get("start_epoch", 0)
    save_every = grpo_cfg.get("save_every_steps", 200)
    num_epochs = num_epochs or config["train"]["epochs"]
    grad_accum = config["train"].get("grad_accum_steps", 1)
    best_val = float("inf")
    best_epoch = 0

    initial_state_dict = None  # CPU snapshot of the pre-GRPO weights, not a second model
    start_epoch = 0
    start_step = 0
    global_step = 0

    resume_meta = load_step_checkpoint(model, optimizer, output_dir, config)
    if resume_meta is not None:
        start_epoch = resume_meta["epoch"]
        start_step = resume_meta["step"]
        global_step = resume_meta["global_step"]
        logging.info(
            "Resuming training at epoch %s, step %s (global_step=%s)",
            start_epoch + 1, start_step, global_step,
        )

    for epoch in range(start_epoch, num_epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        use_grpo = grpo_cfg.get("enabled", False) and epoch >= grpo_start_epoch
        epoch_start_step = start_step if epoch == start_epoch else 0

        if use_grpo and initial_state_dict is None:
            logging.info(
                "Snapshotting initial policy weights (CPU) for GRPO reference at epoch %s",
                epoch + 1,
            )
            # No second model instance is built here (unlike the old build_ref_model
            # + deepcopy approach) — just a CPU-side dict of tensors. GPU memory
            # stays at ~1x model instead of ~2x; the reference forward pass swaps
            # these weights into the live policy model on the fly.
            initial_state_dict = {
                name: param.data.clone().cpu()
                for name, param in model.named_parameters()
            }

        if use_grpo:
            _, global_step = train_one_epoch_grpo(
                model, initial_state_dict, train_loader, optimizer, device,
                epoch, num_epochs, grad_accum, grpo_cfg,
                output_dir=output_dir, config=config,
                start_step=epoch_start_step, global_step=global_step,
                save_every=save_every,
            )
        else:
            train_one_epoch(model, train_loader, optimizer, device, epoch, num_epochs, grad_accum)

        # Epoch finished cleanly — next resume (if any) should start at step 0
        # of the following epoch, not mid-epoch again.
        start_step = 0

        avg_val = validate(model, val_loader, device)
        if is_main_process:
            logging.info("Epoch %s val loss %.4f", epoch + 1, avg_val)

        if is_main_process and avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch + 1
            save_best_checkpoint(model, output_dir, config, epoch=best_epoch, val_loss=best_val)
        distributed_barrier()

    if is_main_process:
        logging.info("Done. Best val loss %.4f at epoch %s", best_val, best_epoch)