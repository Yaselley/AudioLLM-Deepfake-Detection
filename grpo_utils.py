import re
from collections import Counter
import logging

REWARD_WEIGHTS = {
    "label_correct": 1.0,
    "label_incorrect": 0.0,  
    "format": 0.1,
    "reasoning": 0.0,  
}


def configure_rewards(weights: dict) -> None:
    REWARD_WEIGHTS.update({k: v for k, v in weights.items() if k in REWARD_WEIGHTS})


# ============================================================
# Tag extraction -- schema-aware
# ============================================================

def extract_tag(text, tag):
    if text is None:
        return ""
    m = re.search(
        rf"<{tag}>(.*?)</{tag}>",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def extract_label(text):
    if text is None:
        return None
    answer = extract_tag(text, "answer").strip().lower()
    if not answer:
        answer = extract_tag(text, "conclusion").strip().lower()
    if not answer:
        answer = str(text).strip().lower()
    if "bonafide" in answer or "real" in answer:
        return "Real"
    if "spoof" in answer or "fake" in answer:
        return "Fake"
    return None


def extract_reasoning(text):
    reasoning = extract_tag(text, "reasoning")
    if reasoning:
        return reasoning.lower()
    # tagged_bottom_up fallback: combine observations + reasoning_step
    obs = extract_tag(text, "observations")
    step = extract_tag(text, "reasoning_step")
    combined = " ".join(p for p in (obs, step) if p)
    return combined.lower()


def _detected_schema(text):
    text_lower = str(text).lower()
    if "<reasoning>" in text_lower:
        return "reasoning_answer"
    if "<observations>" in text_lower or "<reasoning_step>" in text_lower:
        return "tagged_bottom_up"
    return None


# ============================================================
# Reasoning Reward (token-F1 between predicted and reference reasoning)
# ============================================================
# Off by default (REWARD_WEIGHTS["reasoning"] == 0.0): exact-token overlap
# against a single reference rationale is easy to reward-hack (keyword
# stuffing / partial copying) without a judge to catch it. Enable via
# configure_rewards({"reasoning": ...}) only after spot-checking completions
# for degenerate overlap-maximizing text.

def token_f1(pred_tokens, gt_tokens):
    pred = Counter(pred_tokens)
    gt = Counter(gt_tokens)

    overlap = sum((pred & gt).values())
    if overlap == 0:
        return 0.0

    precision = overlap / max(sum(pred.values()), 1)
    recall = overlap / max(sum(gt.values()), 1)
    return 2 * precision * recall / (precision + recall)


def reasoning_reward(prediction, ground_truth):
    pred = extract_reasoning(prediction)
    gt = extract_reasoning(ground_truth)

    if pred == "" or gt == "":
        return 0.0

    pred_tokens = re.findall(r"\w+", pred)
    gt_tokens = re.findall(r"\w+", gt)
    return token_f1(pred_tokens, gt_tokens)



def format_reward(prediction) -> float:
    text = str(prediction).strip()
    schema = _detected_schema(text)

    if schema == "reasoning_answer":
        reasoning_ok = len(re.findall(r"<reasoning>(.*?)</reasoning>", text, re.IGNORECASE | re.DOTALL)) == 1
        answer_ok = len(re.findall(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)) == 1
        reasoning_nonempty = bool(extract_tag(text, "reasoning").strip())
        answer_nonempty = bool(extract_tag(text, "answer").strip())

        if not answer_ok or not answer_nonempty:
            return 0.0
        if not reasoning_ok or not reasoning_nonempty:
            return 0.33

        ordered = bool(re.match(
            r"^\s*<reasoning>.+?</reasoning>\s*<answer>.+?</answer>\s*$",
            text, re.IGNORECASE | re.DOTALL,
        ))
        return 1.0 if ordered else 0.66

    if schema == "tagged_bottom_up":
        has_obs = bool(extract_tag(text, "observations").strip())
        has_step = bool(extract_tag(text, "reasoning_step").strip())
        has_concl = bool(extract_tag(text, "conclusion").strip())
        has_answer = bool(extract_tag(text, "answer").strip())

        if not has_answer:
            return 0.0
        score = 0.33
        if has_step or has_obs:
            score = 0.66
        if has_answer and has_concl and (has_step or has_obs):
            score = 1.0
        return score

    # no recognizable tags at all
    return 0.0


# ============================================================
# Combined reward
# ============================================================

def compute_reward(prediction_text, ground_truth_text) -> float:
    reward = 0.0

    pred_label = extract_label(prediction_text)
    gt_label = extract_label(ground_truth_text)
    label_correct = pred_label is not None and pred_label == gt_label

    if label_correct:
        reward += REWARD_WEIGHTS["label_correct"]
        reward += REWARD_WEIGHTS["reasoning"] * reasoning_reward(prediction_text, ground_truth_text)
    else:
        reward += REWARD_WEIGHTS["label_incorrect"]

    reward += REWARD_WEIGHTS["format"] * format_reward(prediction_text)

    return float(reward)