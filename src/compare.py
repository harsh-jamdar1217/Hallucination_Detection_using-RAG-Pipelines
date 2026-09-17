from src.retrieve import load_vectorstore, retrieve_chunks
from src.generate import generate_answer
from src.generate_gemini import generate_answer_gemini
from src.trust_score import evaluate_answer


def get_overall_trust(query, answer, context):
    sentence_results = evaluate_answer(query, answer, context)
    if not sentence_results:
        return 1.0, sentence_results
    overall = min(r["trust_score"] for r in sentence_results)
    return overall, sentence_results


def compare_models(query, k=3):
    vectorstore = load_vectorstore()
    chunks = retrieve_chunks(vectorstore, query, k=k)
    context = [c.page_content for c in chunks]

    llama_answer = generate_answer(query, chunks)
    gemini_answer = generate_answer_gemini(query, context)

    llama_trust, llama_details = get_overall_trust(query, llama_answer, context)
    gemini_trust, gemini_details = get_overall_trust(query, gemini_answer, context)

    effective_model = "Llama 3" if llama_trust >= gemini_trust else "Gemini"
    effective_answer = llama_answer if llama_trust >= gemini_trust else gemini_answer

    return {
        "query": query,
        "context": context,
        "llama": {"answer": llama_answer, "trust_score": llama_trust, "details": llama_details},
        "gemini": {"answer": gemini_answer, "trust_score": gemini_trust, "details": gemini_details},
        "effective_model": effective_model,
        "effective_answer": effective_answer
    }


def print_result(result):
    print(f"\n{'='*60}")
    print(f"Query: {result['query']}")
    print(f"{'='*60}\n")

    print(f"--- Llama 3 (trust={result['llama']['trust_score']:.3f}) ---")
    print(result['llama']['answer'])
    llama_flagged = [d for d in result['llama']['details'] if d['flagged']]
    if llama_flagged:
        print(f"⚠️  {len(llama_flagged)} sentence(s) flagged as possible hallucination:")
        for f in llama_flagged:
            print(f"   - {f['sentence']}")
    print()

    print(f"--- Gemini (trust={result['gemini']['trust_score']:.3f}) ---")
    print(result['gemini']['answer'])
    gemini_flagged = [d for d in result['gemini']['details'] if d['flagged']]
    if gemini_flagged:
        print(f"⚠️  {len(gemini_flagged)} sentence(s) flagged as possible hallucination:")
        for f in gemini_flagged:
            print(f"   - {f['sentence']}")
    print()

    print(f"✅ Effective answer (from {result['effective_model']}):")
    print(result['effective_answer'])
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Hallucination Detection Bot — Llama 3 vs Gemini")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("exit", "quit", ""):
            print("Goodbye!")
            break

        try:
            result = compare_models(query)
            print_result(result)
        except Exception as e:
            print(f"Error processing query: {e}\n")