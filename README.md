---
title: AI Document Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: mit
---

# AI Document Chatbot — RAG with Claude

> Upload your documents. Ask questions. Get accurate, cited answers powered by Claude AI and vector search.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io)
[![Claude API](https://img.shields.io/badge/AI-Claude%20API-orange.svg)](https://www.anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Features

- [x] **Multi-format document upload** — PDF, TXT, Markdown, DOCX
- [x] **Streaming responses** — tokens appear as Claude generates them
- [x] **Source attribution** — every answer shows which chunks were used, with relevance scores
- [x] **Vector search** — semantic similarity via sentence-transformers + ChromaDB
- [x] **Multi-session management** — run parallel conversations with shared knowledge base
- [x] **Live cost tracker** — per-session token usage and estimated USD cost
- [x] **Dark-themed UI** — professional GitHub-dark aesthetic built with custom CSS
- [x] **Fully local vector DB** — no ChromaDB server needed, persists to disk
- [x] **Configurable** — all parameters via `.env` (chunk size, top-k, model, temperature)

---

## Architecture

```
                     ┌─────────────────────────────────────────────┐
                     │              User Interface (Streamlit)       │
                     └────────────────────┬────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
     ┌────────▼────────┐        ┌─────────▼──────────┐     ┌────────▼────────┐
     │  Document Upload │        │   Chat Interface    │     │ Session Manager │
     │  (PDF/TXT/MD/   │        │   + Streaming       │     │ (multi-conv)    │
     │   DOCX)         │        │     Response        │     └─────────────────┘
     └────────┬────────┘        └─────────┬──────────┘
              │                           │
     ┌────────▼────────┐        ┌─────────▼──────────┐
     │ document_loader │        │   rag_engine.query  │
     │  (pages + meta) │        │   (top-k chunks)    │
     └────────┬────────┘        └─────────┬──────────┘
              │                           │
     ┌────────▼────────┐        ┌─────────▼──────────┐
     │   TextSplitter   │        │  sentence-         │
     │  (500 tok chunks │        │  transformers      │
     │   50 tok overlap)│        │  (embed query)     │
     └────────┬────────┘        └─────────┬──────────┘
              │                           │
     ┌────────▼───────────────────────────▼──────────┐
     │              ChromaDB (local, on-disk)          │
     │         Cosine similarity vector index          │
     └────────────────────────┬───────────────────────┘
                              │  retrieved context
                     ┌────────▼────────┐
                     │  claude_client   │
                     │  (system prompt  │
                     │  + RAG context   │
                     │  + history)      │
                     └────────┬────────┘
                              │  streaming tokens
                     ┌────────▼────────┐
                     │   Claude API     │
                     │ (claude-sonnet)  │
                     └─────────────────┘
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/JustDreameritis/ai-chatbot-demo.git
cd ai-chatbot-demo

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 5. Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration

All settings are controlled via the `.env` file:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `CHUNK_SIZE` | `500` | Target tokens per document chunk |
| `CHUNK_OVERLAP` | `50` | Token overlap between adjacent chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `MAX_TOKENS` | `4096` | Max tokens in Claude's response |
| `TEMPERATURE` | `0.3` | Generation temperature (0 = factual, 1 = creative) |
| `CHROMA_DB_PATH` | `./chroma_db` | Local path for vector database |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model for embeddings |

---

## How It Works

### 1. Document Ingestion
Uploaded files are processed by `document_loader.py`, which extracts text page-by-page (PDF) or section-by-section (DOCX). Plain text and Markdown are loaded as a single block. Encoding is auto-detected with `chardet`.

### 2. Chunking
`TextSplitter` recursively splits text on paragraph boundaries → sentence boundaries → words, keeping each chunk at or below `CHUNK_SIZE` tokens. Adjacent chunks share `CHUNK_OVERLAP` tokens of context.

### 3. Embedding & Storage
Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors, runs locally on CPU). Embeddings are stored in ChromaDB with a cosine-similarity HNSW index. Chunks are content-hash deduplicated so re-uploading a document is safe.

### 4. Retrieval
When the user asks a question, the query text is embedded with the same model. ChromaDB returns the top-k most similar chunks by cosine similarity. Each chunk includes its source filename, page number, and relevance score.

### 5. Generation
Retrieved chunks are formatted into a structured context block and injected into Claude's system prompt alongside conversation history. Claude is instructed to cite `[Source N]` labels and acknowledge when the documents don't cover a topic. Responses stream token-by-token to the UI.

---

## Project Structure

```
ai-chatbot-demo/
├── app.py               # Streamlit UI — chat, uploads, session management
├── rag_engine.py        # RAG pipeline: chunk, embed, store, retrieve
├── claude_client.py     # Claude API wrapper: streaming, history, cost tracking
├── document_loader.py   # Multi-format document ingestion (PDF/TXT/MD/DOCX)
├── config.py            # .env-based configuration
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore
├── docs/
│   └── SOW-template.md  # Statement of Work for client engagements
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Model | [Claude API](https://www.anthropic.com) (Anthropic) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` |
| Vector DB | [ChromaDB](https://www.trychroma.com/) (local persistent) |
| Web UI | [Streamlit](https://streamlit.io/) |
| PDF parsing | [PyPDF2](https://pypi.org/project/PyPDF2/) |
| DOCX parsing | [python-docx](https://python-docx.readthedocs.io/) |
| Token counting | [tiktoken](https://github.com/openai/tiktoken) |
| Config | [python-dotenv](https://github.com/theskumar/python-dotenv) |

---

## License

MIT — free to use, modify, and deploy. See [LICENSE](LICENSE) for details.

---

## Built by

**JustDreameritis** — AI & Automation Developer

- Upwork: [upwork.com/freelancers/~JustDreameritis](https://www.upwork.com/freelancers/~JustDreameritis)
- GitHub: [github.com/JustDreameritis](https://github.com/JustDreameritis)

> Available for custom AI chatbot development. See [`docs/SOW-template.md`](docs/SOW-template.md) for engagement details.
