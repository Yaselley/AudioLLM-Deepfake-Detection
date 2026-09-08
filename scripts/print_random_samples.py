import json
import random
import re
import sys

def extract_answer(pred: str) -> str:
    """Pull out Real/Fake from the <answer>...</answer> tag."""
    match = re.search(r"<answer>\s*(Real|Fake)\s*</answer>", pred, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "Unknown"

def load_data(path: str):
    """Load either a JSON array (.json) or newline-delimited JSON (.jsonl)."""
    with open(path, "r") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                data = json.load(f)
            else:
                # Not a JSON array — assume JSONL even without the extension
                data = [json.loads(line) for line in f if line.strip()]
    return data

def main(path: str, n: int = 10, seed: int = None):
    data = load_data(path)

    if seed is not None:
        random.seed(seed)

    n = min(n, len(data))
    sample = random.sample(data, n)

    correct = 0
    for i, item in enumerate(sample, 1):
        gt = item.get("ground_truth", "Unknown")
        pred_text = item.get("prediction", "")
        pred_answer = extract_answer(pred_text)
        is_correct = pred_answer.lower() == gt.lower()
        correct += is_correct

        print(f"--- Sample {i} ---")
        print(f"ID: {item.get('id')}")
        print(f"Ground truth: {gt}")
        print(f"Predicted answer: {pred_answer}")
        print(f"Correct: {is_correct}")
        print("Prediction:")
        print(pred_text)
        print()

    accuracy = correct / n * 100
    print(f"Accuracy on {n} samples: {correct}/{n} = {accuracy:.1f}%")

if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "predictions.json"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(json_path, num_samples)