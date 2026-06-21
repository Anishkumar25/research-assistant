# AI Research Assistant — RAG over Research Papers

A Retrieval-Augmented Generation (RAG) system that lets you upload research papers as PDFs and ask natural-language questions about them, with answers grounded in and cited to the actual source text — including comparisons across multiple papers.

Built from scratch (no LangChain/LlamaIndex orchestration layer) to demonstrate a working understanding of each component in a RAG pipeline, rather than gluing together a framework.

## Features

- PDF upload and text extraction (single or multiple documents)
- Semantic chunking and embedding (local, no API cost for this step)
- Vector similarity search via ChromaDB
- Grounded answer generation via Llama 3.3 70B (Groq API), with explicit instructions to avoid hallucination
- Inline source citations — every claim is tagged to the document it came from
- Multi-document comparison with per-document retrieval, so every uploaded paper is guaranteed representation (not just whichever scores highest by similarity)
- A small evaluation harness with a measured baseline accuracy

## Tech stack

- **PDF parsing:** PyMuPDF
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`), 384-dim, runs locally
- **Vector store:** ChromaDB
- **LLM:** Llama 3.3 70B via Groq API
- **UI:** Streamlit

## Architecture
PDF Upload → Text Extraction → Chunking (200 words, 40-word overlap)

→ Local Embedding → ChromaDB Storage (tagged with source document)

→ Query Embedding → Per-Document Retrieval (top-k per doc)

→ Grounded Prompt → Llama 3.3 70B → Cited Answer

## Evaluation

A 10-question evaluation set was built from ground-truth facts in a test paper, covering the abstract, methodology, dataset, and results sections. Baseline accuracy: **70%** (7/10 correct).

Diagnosis of the 3 incorrect answers showed two distinct failure modes:
1. **Retrieval misses** — the chunk containing the specific answer ranked outside the top-k and was never seen by the model.
2. **Generation instability** — when context was insufficient, the model sometimes generated unrelated content instead of cleanly admitting it didn't know.

Fixes attempted: increased retrieval depth (`top_k` 3→5), set `temperature=0` for more deterministic output, and tightened the prompt with an explicit fallback instruction for missing information. Result: overall accuracy held at 70%, but the failure mode shifted — a notable finding in itself, suggesting retrieval quality (not just prompt strictness) is the main lever for further improvement.

## Setup

```bash
git clone <your-repo-url>
cd research-assistant
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file with:
GROQ_API_KEY=your_key_here

Run:
```bash
streamlit run app.py
```

## Limitations & future work

- Retrieval is purely embedding-based; no hybrid keyword search or reranking yet
- Chunking is fixed-size by word count, not semantically aware (e.g. doesn't respect section/paragraph boundaries)
- Evaluation set is small (10 questions, single paper); a larger, multi-paper eval set would give more reliable accuracy estimates
- No support for non-text PDF content (figures, tables are not parsed)