from sentence_transformers import SentenceTransformer
from chunk_text import chunk_text
from extract_text import extract_text_from_pdf

model = SentenceTransformer('all-MiniLM-L6-v2')

if __name__ == "__main__":
    text = extract_text_from_pdf("data/Test.pdf")
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First 10 numbers of the first chunk's embedding: {embeddings[0][:10]}")