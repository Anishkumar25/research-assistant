import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import fitz
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()
st.set_page_config(page_title="AI Research Assistant", page_icon="📄")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

model = load_embedding_model()
client = get_groq_client()

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text

def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks

def build_vector_store(chunks_with_sources):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="session_papers")

    texts = [c[0] for c in chunks_with_sources]
    sources = [c[1] for c in chunks_with_sources]

    embeddings = model.encode(texts).tolist()
    ids = [f"chunk_{i}" for i in range(len(texts))]
    metadatas = [{"source": s} for s in sources]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return collection

def retrieve_relevant_chunks(collection, query, sources, top_k_per_doc=3):
    # Retrieve top_k_per_doc chunks from EACH document separately, using a metadata filter,
    # so every uploaded paper is guaranteed representation — not just whichever scores highest overall
    query_embedding = model.encode([query]).tolist()
    all_chunks = []
    all_metadatas = []
    for source in sources:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k_per_doc,
            where={"source": source}
        )
        all_chunks.extend(results["documents"][0])
        all_metadatas.extend(results["metadatas"][0])
    return all_chunks, all_metadatas

def generate_answer(collection, question, sources, top_k_per_doc=3):
    chunks, metadatas = retrieve_relevant_chunks(collection, question, sources, top_k_per_doc)
    context = "\n\n".join([
        f"[Source: {metadatas[i]['source']}]\n{chunks[i]}" for i in range(len(chunks))
    ])

    prompt = f"""You are a research assistant. Answer the question using ONLY the context below.

Each context piece is labeled with its source document. When you state a fact, cite which document it came from, like this: (Source: filename.pdf). If comparing multiple documents, clearly organize the comparison by document.

Rules:
- If facts relevant to the question are present in the context, answer using them — for comparison questions, synthesize a comparison by summarizing what each document's own context independently says, even if no single chunk explicitly compares them.
- Only respond with exactly "The provided context does not contain this information." if NONE of the context relates to the question at all.
- Always cite which document each piece of information came from.
- Never generate unrelated questions, quizzes, or content outside of directly answering the question asked.
Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=700,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, chunks, metadatas

st.title("📄 AI Research Assistant")
st.write("Upload one or more research papers and ask questions — including comparisons across papers.")

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    current_filenames = sorted([f.name for f in uploaded_files])
    if "filenames" not in st.session_state or st.session_state["filenames"] != current_filenames:
        with st.spinner("Processing PDFs..."):
            chunks_with_sources = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                chunks = chunk_text(text)
                for c in chunks:
                    chunks_with_sources.append((c, file.name))
            st.session_state["collection"] = build_vector_store(chunks_with_sources)
            st.session_state["filenames"] = current_filenames
        st.success(f"Processed {len(chunks_with_sources)} chunks from {len(uploaded_files)} document(s)")

    question = st.text_input("Ask a question (try comparing the papers if you uploaded more than one):")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            answer, used_chunks, used_metadatas = generate_answer(
                st.session_state["collection"], question, st.session_state["filenames"]
            )
        st.markdown("### Answer")
        st.write(answer)
        with st.expander("Show source chunks used"):
            for i, chunk in enumerate(used_chunks):
                st.markdown(f"**From {used_metadatas[i]['source']}:**")
                st.write(chunk)