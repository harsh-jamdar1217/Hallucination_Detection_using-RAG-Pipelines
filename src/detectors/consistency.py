import ollama

def generate_multiple_answers(query, context, n=5, model="llama3", temperature=0.7):
    context_text = context if isinstance(context, str) else "\n".join(context)

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I don't know based on the given context."

Context:
{context_text}

Question: {query}

Answer:"""

    answers = []
    for _ in range(n):
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature}
        )
        answers.append(response['response'].strip())
    return answers

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def consistency_score(answers):
    if len(answers) < 2:
        return 1.0

    embeddings = model_embed.encode(answers)
    sim_matrix = cosine_similarity(embeddings)

    n = len(answers)
    pairwise_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_scores.append(sim_matrix[i][j])

    return float(np.mean(pairwise_scores))

def check_self_consistency(query, context, n=5, model="llama3"):
    answers = generate_multiple_answers(query, context, n=n, model=model)
    score = consistency_score(answers)
    return {"answers": answers, "consistency_score": score}

if __name__ == "__main__":
    context = ["The Statue of Liberty is made of copper. It was a gift from France, completed in 1886."]
    query = "What is the Statue of Liberty made of, and when was it completed?"

    result = check_self_consistency(query, context)

    print(f"Consistency score: {result['consistency_score']:.3f}\n")
    for i, ans in enumerate(result['answers'], 1):
        print(f"--- Answer {i} ---")
        print(ans)
        print()