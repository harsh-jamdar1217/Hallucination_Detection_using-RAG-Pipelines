def compute_trust_score(similarity_score, judge_score, consistency_score,
                         w_similarity=0.3, w_judge=0.5, w_consistency=0.2):
    trust = (
        w_similarity * similarity_score +
        w_judge * judge_score +
        w_consistency * consistency_score
    )
    return trust

from src.detectors.similarity import detect_hallucinated_sentences
from src.detectors.judge import judge_sentences
from src.detectors.consistency import check_self_consistency

def evaluate_answer(query, answer, context, threshold=0.5):
    sim_results = detect_hallucinated_sentences(answer, context, threshold=threshold)
    judge_results = judge_sentences(answer, context)
    consistency_result = check_self_consistency(query, context)
    consistency = consistency_result["consistency_score"]

    combined = []
    for sim_item, judge_item in zip(sim_results, judge_results):
        trust = compute_trust_score(
            similarity_score=sim_item["similarity"],
            judge_score=judge_item["judge_score"],
            consistency_score=consistency
        )
        combined.append({
            "sentence": sim_item["sentence"],
            "similarity": sim_item["similarity"],
            "judge_score": judge_item["judge_score"],
            "consistency_score": consistency,
            "trust_score": trust,
            "flagged": trust < threshold
        })
    return combined

if __name__ == "__main__":
    context = ["The Statue of Liberty is made of copper. It was a gift from France, completed in 1886."]
    query = "What is the Statue of Liberty made of, and who designed it?"
    answer = "The Statue of Liberty is made of copper. It was designed by Napoleon Bonaparte himself."

    results = evaluate_answer(query, answer, context)

    for r in results:
        flag = "🚩 FLAGGED" if r["flagged"] else "✅ ok"
        print(f"{flag} trust={r['trust_score']:.3f} (sim={r['similarity']:.2f}, judge={r['judge_score']:.2f}, consist={r['consistency_score']:.2f})")
        print(f"   → {r['sentence']}\n")