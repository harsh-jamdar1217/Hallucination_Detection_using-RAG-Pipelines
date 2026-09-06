import os
import time
from google import genai
from google.genai import errors

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def build_prompt(query, chunks):
    context = chunks if isinstance(chunks, str) else "\n\n".join(chunks)
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I don't know based on the given context."

Context:
{context}

Question: {query}

Answer:"""
    return prompt


def generate_answer_gemini(query, chunks, model="gemini-3.6-flash", max_retries=3):
    prompt = build_prompt(query, chunks)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text.strip()
        except errors.ServerError as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"Server busy, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise


if __name__ == "__main__":
    context = ["The Statue of Liberty is made of copper. It was a gift from France, completed in 1886."]
    query = "What is the Statue of Liberty made of, and when was it completed?"

    answer = generate_answer_gemini(query, context)
    print(f"Query: {query}\n")
    print(f"Gemini Answer: {answer}")