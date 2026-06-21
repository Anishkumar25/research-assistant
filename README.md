# 📄 AI Research Assistant — Retrieval-Augmented Generation for Research Papers

> Upload research papers, ask natural-language questions, and get grounded, cited answers — including comparisons across multiple papers. Built from scratch to demonstrate a real understanding of every layer of a RAG pipeline, not a framework wrapped around an API call.

**[Live Demo](https://researchassistant2512.streamlit.app/)** · **[Source Code](https://github.com/Anishkumar25/research-assistant/)**

---

## Why this project is different

Most "RAG chatbot" projects stop at "upload a PDF, ask a question, get an answer." This one goes further:

- **It's measured, not just demoed.** A 10-question evaluation set with known ground-truth answers gives a concrete baseline accuracy (70%), not a vague "it works great."
- **Failure modes were diagnosed, not papered over.** Incorrect answers were traced to two distinct root causes — retrieval misses vs. generation instability — which require different fixes, and that diagnosis is documented below.
- **Multi-document comparison actually works correctly.** Naive pooled retrieval let one document dominate the context for cross-paper questions; this was caught, diagnosed, and fixed with per-document metadata filtering (see "Key design decisions").
- **No orchestration framework.** Built directly with PyMuPDF, sentence-transformers, and ChromaDB instead of LangChain/LlamaIndex — every step of the pipeline (chunking, embedding, retrieval, prompting) is explicit and auditable, not hidden behind abstraction layers.

## Features

- 📄 Multi-PDF upload and text extraction
- ✂️ Configurable chunking with overlap to preserve cross-boundary context
- 🧠 Local semantic embeddings (no API cost for this step)
- 🔍 Vector similarity search via ChromaDB, with per-document metadata filtering for balanced multi-document retrieval
- 💬 Grounded answer generation via Llama 3.3 70B (Groq API), explicitly instructed to avoid hallucination
- 📌 Inline source citations — every claim is tagged to the exact document it came from
- 🔀 Cross-document comparison and synthesis
- 📊 A built-in evaluation harness with a documented accuracy baseline

## Architecture
PDF(s) Upload

│

▼

Text Extraction (PyMuPDF)

│

▼

Chunking (200 words, 40-word overlap)

│

▼

Local Embedding (sentence-transformers, all-MiniLM-L6-v2, 384-dim)

│

▼

Vector Storage (ChromaDB, chunks tagged with source document)

│

▼

Query → Per-Document Retrieval (top-k per document, metadata-filtered)

│

▼

Grounded Prompt Construction (labeled, cited context)

│

▼

Llama 3.3 70B (Groq API) → Cited, Grounded Answer

## Tech stack

| Layer | Technology |
|---|---|
| PDF parsing | PyMuPDF |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) — local, free, 384-dim |
| Vector store | ChromaDB |
| LLM | Llama 3.3 70B via Groq API (free tier, no orchestration framework) |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

## Evaluation

A 10-question evaluation set was constructed from ground-truth facts spanning the abstract, methodology, dataset, and results sections of a test paper.

**Baseline accuracy: 70% (7/10 correct)**

Manual diagnosis of incorrect answers revealed two distinct failure modes:

1. **Retrieval misses** — the chunk containing the specific answer ranked outside the top-k similarity results and was never shown to the model, even though it existed in the document.
2. **Generation instability** — when the retrieved context didn't contain the answer, the model occasionally generated unrelated content (e.g. an invented multiple-choice quiz) instead of cleanly admitting it didn't know.

**Fixes applied:** increased retrieval depth (`top_k` 3 → 5), set `temperature=0` for deterministic output, and added an explicit, strict fallback instruction in the prompt for missing information.

**Result:** overall accuracy held at 70%, but the *distribution* of errors shifted — suggesting retrieval quality, not prompt strictness alone, is the primary lever for further improvement. This is documented as-is rather than smoothed over, since the iteration process is the actual evidence of engineering rigor.

## Key design decisions

- **In-memory vector store per session, not persistent.** The deployed app uses a fresh in-memory ChromaDB collection per session rather than persisting to disk, avoiding ID-collision errors on re-upload and keeping each user's documents isolated.
- **Per-document retrieval for multi-doc queries.** Initial testing showed that pooling all chunks together and taking a global top-k let one document dominate comparison answers entirely, even when the question explicitly asked about multiple papers. Switching to per-document metadata-filtered retrieval (querying each source separately, then combining) guarantees every uploaded document is represented.
- **Local embeddings, API-based generation.** Embedding generation runs locally (free, fast, no rate limits) while only the final generation step calls an external API — minimizing cost and external dependency for the most frequently-run part of the pipeline.

## Setup

```bash
git clone <your-repo-url>
cd research-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

- **GROQ_API_KEY =**your_key_here
Run locally:
```bash
streamlit run app.py
```

## Project structure
research-assistant/

├── app.py              # Main Streamlit application

├── extract_text.py     # PDF text extraction (standalone/testing)

├── chunk_text.py        # Text chunking logic (standalone/testing)

├── embed_chunks.py      # Embedding generation (standalone/testing)

├── store_chunks.py      # Vector storage (standalone/testing)

├── retrieve.py          # Semantic retrieval (standalone/testing)

├── generate_answer.py   # End-to-end RAG pipeline (standalone/testing)

├── eval.py              # Evaluation harness

├── eval_set.json        # Ground-truth Q&A pairs for evaluation

└── requirements.txt

## Limitations & future work

- Retrieval is purely embedding-based; no hybrid keyword search or reranking layer yet
- Chunking is fixed-size by word count, not semantically or section-aware
- The evaluation set is small (10 questions, single source paper); a larger, multi-paper evaluation set would give more statistically reliable accuracy estimates
- Figures, tables, and equations in PDFs are not parsed beyond their raw extracted text
- No persistent multi-session storage — each session starts fresh
