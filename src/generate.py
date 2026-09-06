def build_prompt(query, chunks):
    context = "\n\n".join([chunk.page_content for chunk in chunks])
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I don't know based on the given context."

Context:
{context}

Question: {query}

Answer:"""
    return prompt

import ollama

def generate_answer(query, chunks, model="llama3"):
    prompt = build_prompt(query, chunks)
    response = ollama.generate(model=model, prompt=prompt)
    return response['response']

from src.retrieve import load_vectorstore, retrieve_chunks

if __name__ == "__main__":
    vectorstore = load_vectorstore()
    query = "What is the Statue of Liberty made of?"

    chunks = retrieve_chunks(vectorstore, query)
    answer = generate_answer(query, chunks)

    print(f"\nQuery: {query}\n")
    print(f"Answer: {answer}\n")
    print("--- Retrieved context used ---")
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}] {chunk.page_content[:150]}...")