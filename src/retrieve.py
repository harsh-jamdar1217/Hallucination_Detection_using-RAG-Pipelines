from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_vectorstore(persist_directory="chroma_db"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    return vectorstore

def retrieve_chunks(vectorstore, query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    vectorstore = load_vectorstore()
    query = "What is the Statue of Liberty made of?"   # replace with a question relevant to your docs
    chunks = retrieve_chunks(vectorstore, query)

    print(f"\nQuery: {query}\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} (source: {chunk.metadata.get('source')}) ---")
        print(chunk.page_content)
        print()