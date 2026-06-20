from extract_text import extract_text_from_pdf

def chunk_text(text, chunk_size=200, overlap=40):
    # chunk_size and overlap are measured in words here, simplest to reason about
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # step forward, but re-include the last bit of overlap
    return chunks

if __name__ == "__main__":
    text = extract_text_from_pdf("data/Test.pdf")
    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")
    print("\n=== Chunk 1 ===")
    print(chunks[0])
    print("\n=== Chunk 2 ===")
    print(chunks[1])