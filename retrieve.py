import chromadb
from embed_chunks import model

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="research_papers")

def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0]  # the top_k chunk texts, closest first

if __name__ == "__main__":
    question = "What is the main contribution of this paper?"  # swap this for a question relevant to your actual PDF
    top_chunks = retrieve_relevant_chunks(question)

    for i, chunk in enumerate(top_chunks):
        print(f"\n=== Retrieved chunk {i+1} ===")
        print(chunk[:300])