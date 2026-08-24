from __future__ import annotations

import copy
import logging
import time

import torch
from torch.utils.data import DataLoader

from grpo_utils import compute_reward
from train_common import batch_to_samples, build_model


def build_ref_model(model_cfg: dict, policy_model, device: torch.device):
    ref_model = build_model(model_cfg)
    ref_model.load_state_dict(copy.deepcopy(policy_model.state_dict()))
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    return ref_model


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
        eos_count = eos.cumsum(dim=1)
        after_first_eos = eos_count > 1
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
    # Compute log softmax over vocabulary
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Gather log probs for the actual tokens
    # labels shape: (batch_size, seq_len)
    # We need to select log_probs[i, j, labels[i, j]] for each i, j
    
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for gathering
    # log_probs: (batch_size * seq_len, vocab_size)
    log_probs_flat = log_probs.view(-1, vocab_size)
    
    # labels: (batch_size * seq_len)
    labels_flat = labels.view(-1)
    
    # Gather: (batch_size * seq_len)
    selected_log_probs = log_probs_flat.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
    
    # Reshape back: (batch_size, seq_len)
    selected_log_probs = selected_log_probs.view(batch_size, seq_len)
    
    return selected_log_probs


def grpo_train_step(policy_model, ref_model, samples, grpo_cfg, *, step=0):
    generate_cfg = _default_generate_cfg(grpo_cfg)
    num_rollouts = int(grpo_cfg.get("num_rollouts", 4))
    beta = grpo_cfg.get("kl_beta", 0.0)
    clip_ratio = grpo_cfg.get("clip_ratio", 0.2)
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
        ref_logits = ref_model.compute_logits_for_completions(expanded, completion_ids)
        ref_logprobs = get_log_probs(ref_logits, completion_ids)

        # Advantages
        reward_matrix = rewards.view(batch_size, num_rollouts).transpose(0, 1)
        reward_mean = reward_matrix.mean(dim=0, keepdim=True)
        reward_std = reward_matrix.std(dim=0, keepdim=True).clamp_min(1e-6)
        advantages = ((reward_matrix - reward_mean) / reward_std).transpose(0, 1).reshape(-1)
        advantages = advantages.detach()

    policy_model.train()
    logits = policy_model.compute_logits_for_completions(expanded, completion_ids)
    log_probs = get_log_probs(logits, completion_ids)

    masked_log_probs = log_probs * completion_mask
    masked_ref_logprobs = ref_logprobs * completion_mask

    # Non-negative KL (Option A)
    log_ratio = masked_log_probs - masked_ref_logprobs
    kl_per_token = torch.exp(log_ratio) - log_ratio - 1
    kl_term = kl_per_token.sum(dim=1).mean()

    ratio = torch.exp(masked_log_probs - masked_ref_logprobs.detach())
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -torch.min(
        ratio * advantages.unsqueeze(1),
        clipped_ratio * advantages.unsqueeze(1)
    ).sum(dim=1).mean()

    loss = policy_loss + beta * kl_term

    if step % log_freq == 0:
        logging.info(
            "[GRPO] step %d | loss=%.4f | kl=%.4f | reward_mean=%.4f | reward_std=%.4f | "
            "ratio_mean=%.2f | ratio_max=%.2f",
            step, loss.item(), kl_term.item(),
            rewards.mean().item(), rewards.std().item(),
            ratio.mean().item(), ratio.max().item(),
        )

    return loss


def train_one_epoch_grpo(
    policy_model,
    ref_model,
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

        loss = grpo_train_step(policy_model, ref_model, samples, grpo_cfg, step=step) / grad_accum_steps
        loss.backward()
        total_loss += loss.item() * grad_accum_steps
        steps_done += 1
        global_step += 1

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % grpo_cfg.get("log_freq", 50) == 0:
            logging.info(
                "[GRPO] Epoch %s/%s | step %s/%s | loss %.4f",
                epoch + 1, num_epochs, step, len(train_loader), loss.item() * grad_accum_steps,
            )

        if save_every > 0 and (step + 1) % save_every == 0:
            save_step_checkpoint(
                policy_model, optimizer, output_dir, config,
                epoch=epoch, step=step + 1, global_step=global_step,
            )

    if steps_done > 0 and (step + 1) % grad_accum_steps != 0:
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

    grpo_cfg = config["train"].get("grpo", {})
    grpo_start_epoch = grpo_cfg.get("start_epoch", 0)
    save_every = grpo_cfg.get("save_every_steps", 200)
    num_epochs = num_epochs or config["train"]["epochs"]
    grad_accum = config["train"].get("grad_accum_steps", 1)
    best_val = float("inf")
    best_epoch = 0

    ref_model = None
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
        use_grpo = grpo_cfg.get("enabled", False) and epoch >= grpo_start_epoch
        epoch_start_step = start_step if epoch == start_epoch else 0

        if use_grpo and ref_model is None:
            logging.info("Building frozen reference policy for GRPO at epoch %s", epoch + 1)
            ref_model = build_ref_model(config["model"], model, device)

        if use_grpo:
            _, global_step = train_one_epoch_grpo(
                model, ref_model, train_loader, optimizer, device,
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
        logging.info("Epoch %s val loss %.4f", epoch + 1, avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch + 1
            save_best_checkpoint(model, output_dir, config, epoch=best_epoch, val_loss=best_val)

    logging.info("Done. Best val loss %.4f at epoch %s", best_val, best_epoch)