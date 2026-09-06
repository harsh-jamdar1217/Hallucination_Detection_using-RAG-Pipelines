import re

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def sentence_similarity_scores(answer, context_chunks):
    sentences = split_into_sentences(answer)
    if not sentences:
        return []

    context_text = context_chunks if isinstance(context_chunks, str) else "\n".join(context_chunks)

    sentence_embeddings = model.encode(sentences)
    context_embedding = model.encode([context_text])

    similarities = cosine_similarity(sentence_embeddings, context_embedding)

    results = []
    for sentence, score in zip(sentences, similarities[:, 0]):
        results.append({"sentence": sentence, "similarity": float(score)})
    return results

def detect_hallucinated_sentences(answer, context_chunks, threshold=0.5):
    scored = sentence_similarity_scores(answer, context_chunks)
    for item in scored:
        item["flagged"] = item["similarity"] < threshold
    return scored

if __name__ == "__main__":
    context = ["The Statue of Liberty is made of copper. It was a gift from France, completed in 1886."]
    answer = "The Statue of Liberty is made of copper. It was designed by Napoleon Bonaparte himself."

    results = detect_hallucinated_sentences(answer, context)
    for r in results:
        flag = "🚩 FLAGGED" if r["flagged"] else "✅ ok"
        print(f"{flag} (sim={r['similarity']:.3f}): {r['sentence']}")