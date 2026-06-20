import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import fitz
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()
st.set_page_config(page_title="AI Research Assistant", page_icon="📄")

# @st.cache_resource means these only load ONCE, not on every interaction —
# without this, Streamlit would reload the embedding model every time you click anything
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

def build_vector_store(chunks):
    chroma_client = chromadb.Client()  # in-memory: fresh per session, no leftover-ID errors
    collection = chroma_client.get_or_create_collection(name="session_papers")
    embeddings = model.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks)
    return collection

def retrieve_relevant_chunks(collection, query, top_k=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0]

def generate_answer(collection, question, top_k=3):
    chunks = retrieve_relevant_chunks(collection, question, top_k)
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""You are a research assistant. Answer the question using ONLY the context below. If the answer isn't in the context, say so clearly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, chunks

st.title("📄 AI Research Assistant")
st.write("Upload a research paper and ask questions about it.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if "filename" not in st.session_state or st.session_state["filename"] != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(text)
            st.session_state["collection"] = build_vector_store(chunks)
            st.session_state["filename"] = uploaded_file.name
        st.success(f"Processed {len(chunks)} chunks from {uploaded_file.name}")

    question = st.text_input("Ask a question about the paper:")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            answer, used_chunks = generate_answer(st.session_state["collection"], question)
        st.markdown("### Answer")
        st.write(answer)
        with st.expander("Show source chunks used"):
            for i, chunk in enumerate(used_chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(chunk)