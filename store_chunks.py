import chromadb
from embed_chunks import model
from chunk_text import chunk_text
from extract_text import extract_text_from_pdf

# This saves the database to a folder called "chroma_db" on disk,
# so it persists even after the script finishes running
client = chromadb.PersistentClient(path="chroma_db")

# A "collection" is basically a table — this creates it if it doesn't exist yet
collection = client.get_or_create_collection(name="research_papers")

if __name__ == "__main__":
    text = extract_text_from_pdf("data/Test.pdf")
    chunks = chunk_text(text)
    embeddings = model.encode(chunks).tolist()  # Chroma wants plain lists, not numpy arrays

    ids = [f"chunk_{i}" for i in range(len(chunks))]  # every chunk needs a unique ID

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks  # store the raw text too, so we can read it back later
    )

    print(f"Stored {collection.count()} chunks in the database.")