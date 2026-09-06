import ollama
import re
from src.detectors.similarity import split_into_sentences

def build_judge_prompt(sentence, context):
    prompt = f"""You are a strict fact-checker. Your job is to check if a SENTENCE is fully supported by the given CONTEXT.

Context:
{context}

Sentence to check:
{sentence}

Rate how well the sentence is supported by the context, on a scale from 0.0 to 1.0:
- 1.0 = fully supported, every claim in the sentence is stated in the context
- 0.5 = partially supported, or a reasonable inference from the context
- 0.0 = not supported at all, contradicts the context, or is fabricated

Respond with ONLY a number between 0.0 and 1.0. No explanation, no other text."""
    return prompt

def get_judge_score(sentence, context, model="llama3"):
    prompt = build_judge_prompt(sentence, context)
    response = ollama.generate(model=model, prompt=prompt)
    raw_output = response['response'].strip()

    match = re.search(r'(\d+\.?\d*)', raw_output)
    if match:
        score = float(match.group(1))
        score = max(0.0, min(1.0, score))
        return score
    else:
        print(f"Warning: couldn't parse judge output: '{raw_output}' — defaulting to 0.5")
        return 0.5
    
    from detectors.similarity import split_into_sentences

def judge_sentences(answer, context_chunks, model="llama3"):
    sentences = split_into_sentences(answer)
    context_text = context_chunks if isinstance(context_chunks, str) else "\n".join(context_chunks)

    results = []
    for sentence in sentences:
        score = get_judge_score(sentence, context_text, model=model)
        results.append({"sentence": sentence, "judge_score": score})
    return results


if __name__ == "__main__":
    context = ["The Statue of Liberty is made of copper. It was a gift from France, completed in 1886."]
    answer = "The Statue of Liberty is made of copper. It was designed by Napoleon Bonaparte himself."

    results = judge_sentences(answer, context)
    for r in results:
        flag = "🚩 FLAGGED" if r["judge_score"] < 0.5 else "✅ ok"
        print(f"{flag} (judge_score={r['judge_score']:.2f}): {r['sentence']}")