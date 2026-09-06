from src.trust_score import evaluate_answer
from src.load_ragtruth import load_ragtruth, get_binary_label
import json
import os
import time


def example_trust_result(query, answer, context, threshold=0.5):
    sentence_results = evaluate_answer(query, answer, context, threshold=threshold)
    if not sentence_results:
        return {"predicted_hallucinated": 0, "min_trust_score": 1.0, "sentence_results": []}

    min_trust = min(r["trust_score"] for r in sentence_results)
    predicted_hallucinated = 1 if min_trust < threshold else 0

    return {
        "predicted_hallucinated": predicted_hallucinated,
        "min_trust_score": min_trust,
        "sentence_results": sentence_results
    }


def run_evaluation(num_examples=None, checkpoint_path="eval_results.json", checkpoint_every=10):
    dataset = load_ragtruth()
    test_set = dataset["test"]
    if num_examples:
        test_set = test_set.select(range(min(num_examples, len(test_set))))

    # Resume from checkpoint if it exists
    results = []
    start_idx = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx} examples already done")

    total = len(test_set)
    for i in range(start_idx, total):
        example = test_set[i]
        query = example["query"]
        answer = example["output"]
        context = example["context"]
        true_label = get_binary_label(example)

        try:
            result = example_trust_result(query, answer, context)
            results.append({
                "id": example["id"],
                "true_label": true_label,
                "predicted_label": result["predicted_hallucinated"],
                "min_trust_score": result["min_trust_score"]
            })
            print(f"[{i+1}/{total}] id={example['id']} true={true_label} pred={result['predicted_hallucinated']} trust={result['min_trust_score']:.3f}")
        except Exception as e:
            print(f"[{i+1}/{total}] Error on example id={example['id']}: {e}")
            results.append({
                "id": example["id"],
                "true_label": true_label,
                "predicted_label": None,
                "min_trust_score": None,
                "error": str(e)
            })

        if (i + 1) % checkpoint_every == 0 or (i + 1) == total:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"--- checkpoint saved at {i+1}/{total} ---")

    return results

def compute_metrics(results, threshold=None):
    valid = [r for r in results if r["predicted_label"] is not None]

    if threshold is not None:
        # Re-derive predictions at a custom threshold using min_trust_score
        y_pred = [1 if r["min_trust_score"] < threshold else 0 for r in valid]
    else:
        y_pred = [r["predicted_label"] for r in valid]

    y_true = [r["true_label"] for r in valid]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "threshold": threshold if threshold is not None else 0.5,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def sweep_thresholds(results, thresholds=None):
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10}")
    for t in thresholds:
        m = compute_metrics(results, threshold=t)
        print(f"{t:<10.2f} {m['precision']:<10.3f} {m['recall']:<10.3f} {m['f1']:<10.3f} {m['accuracy']:<10.3f}")


if __name__ == "__main__":
    print("Running full evaluation on RAGTruth test set (2700 examples)...")
    print("This will take several hours. You can stop anytime (Ctrl+C) and re-run to resume from checkpoint.\n")
    results = run_evaluation(num_examples=None, checkpoint_path="eval_results.json")
    print(f"\nCompleted {len(results)} examples\n")

    print("Threshold sweep on full results:")
    sweep_thresholds(results)