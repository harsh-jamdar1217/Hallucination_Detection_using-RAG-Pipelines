import json
import os


def load_results(path):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping.")
        return None
    with open(path, "r") as f:
        return json.load(f)


def compute_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "n": len(y_true),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def print_table(title, metrics):
    print(f"\n{'='*50}")
    print(title)
    print(f"{'='*50}")
    print(f"{'Metric':<15}{'Value':<10}")
    print(f"{'Precision':<15}{metrics['precision']:<10}")
    print(f"{'Recall':<15}{metrics['recall']:<10}")
    print(f"{'F1-score':<15}{metrics['f1']:<10}")
    print(f"{'Accuracy':<15}{metrics['accuracy']:<10}")
    print(f"{'Samples (n)':<15}{metrics['n']:<10}")
    print(f"(TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']})")


# ============================================================
# 1. RAGTruth final evaluation (from eval_results.json)
# ============================================================
ragtruth_results = load_results("eval_results.json")
if ragtruth_results:
    valid = [r for r in ragtruth_results if r.get("predicted_label") is not None]
    y_true = [r["true_label"] for r in valid]
    y_pred = [r["predicted_label"] for r in valid]
    metrics = compute_metrics(y_true, y_pred)
    print_table("RAGTRUTH FINAL EVALUATION (Trust-Score System)", metrics)


# ============================================================
# 2. Llama 3 vs Gemini comparison (from eval_compare_results.json)
# ============================================================
compare_results = load_results("eval_compare_results.json")
if compare_results:
    for model_key, label in [("llama", "LLAMA 3"), ("gemini", "GEMINI")]:
        valid = [r for r in compare_results if r.get(model_key, {}).get("predicted_hallucinated") is not None]
        y_true = [r["true_label"] for r in valid]
        y_pred = [r[model_key]["predicted_hallucinated"] for r in valid]
        metrics = compute_metrics(y_true, y_pred)
        print_table(f"{label} — HALLUCINATION DETECTION PERFORMANCE", metrics)