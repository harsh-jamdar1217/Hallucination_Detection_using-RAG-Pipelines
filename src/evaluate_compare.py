import json
import os
import time
from src.load_ragtruth import load_ragtruth, get_binary_label
from src.generate_gemini import generate_answer_gemini, generate_gemini_for_consistency
from src.trust_score import evaluate_answer
import ollama


def generate_llama_for_eval(query, context):
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I don't know based on the given context."

Context:
{context}

Question: {query}

Answer:"""
    response = ollama.generate(model="llama3", prompt=prompt)
    return response['response'].strip()


def generate_llama_for_consistency(query, context, temperature=0.7):
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I don't know based on the given context."

Context:
{context}

Question: {query}

Answer:"""
    response = ollama.generate(model="llama3", prompt=prompt, options={"temperature": temperature})
    return response['response'].strip()


def get_model_result(query, answer, context, generate_fn, threshold=0.5):
    sentence_results = evaluate_answer(query, answer, context, threshold=threshold, consistency_generate_fn=generate_fn)
    if not sentence_results:
        return {"predicted_hallucinated": 0, "min_trust_score": 1.0}
    min_trust = min(r["trust_score"] for r in sentence_results)
    return {"predicted_hallucinated": 1 if min_trust < threshold else 0, "min_trust_score": min_trust}


def run_comparison_evaluation(num_examples=500, checkpoint_path="eval_compare_results.json", checkpoint_every=10):
    dataset = load_ragtruth()
    test_set = dataset["test"].select(range(num_examples))

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
        context = example["context"]
        true_label = get_binary_label(example)

        entry = {"id": example["id"], "true_label": true_label}

        try:
            llama_answer = generate_llama_for_eval(query, context)
            llama_result = get_model_result(query, llama_answer, context, generate_llama_for_consistency)
            entry["llama"] = llama_result
        except Exception as e:
            entry["llama"] = {"predicted_hallucinated": None, "min_trust_score": None, "error": str(e)}

        try:
            gemini_answer = generate_answer_gemini(query, context)
            gemini_result = get_model_result(query, gemini_answer, context, generate_gemini_for_consistency)
            entry["gemini"] = gemini_result
        except Exception as e:
            entry["gemini"] = {"predicted_hallucinated": None, "min_trust_score": None, "error": str(e)}

        results.append(entry)
        print(f"[{i+1}/{total}] id={example['id']} true={true_label} "
              f"llama={entry['llama'].get('predicted_hallucinated')} "
              f"gemini={entry['gemini'].get('predicted_hallucinated')}")

        if (i + 1) % checkpoint_every == 0 or (i + 1) == total:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"--- checkpoint saved at {i+1}/{total} ---")

        time.sleep(4)

    return results


def compute_model_metrics(results, model_key, threshold=0.5):
    valid = [r for r in results if r.get(model_key, {}).get("min_trust_score") is not None]
    y_true = [r["true_label"] for r in valid]
    y_pred = [1 if r[model_key]["min_trust_score"] < threshold else 0 for r in valid]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "n": len(valid)}


if __name__ == "__main__":
    print("Running dry run on 5 examples first...")
    results = run_comparison_evaluation(num_examples=5, checkpoint_path="eval_compare_dry_run.json")

    print("\n--- Llama metrics ---")
    print(compute_model_metrics(results, "llama"))
    print("\n--- Gemini metrics ---")
    print(compute_model_metrics(results, "gemini"))