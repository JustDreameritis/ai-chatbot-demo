# Case Study: AI Document Chatbot with RAG

## Overview

Production-ready Streamlit application for document Q&A using Retrieval Augmented Generation (RAG). Upload PDFs, Word docs, or text files and chat with Claude to get answers grounded in your documents. Features streaming responses, source attribution, and session management.

## Technical Implementation

### Architecture

```
ai-chatbot-demo/
├── app.py                  # Streamlit web interface
├── config.py               # Pydantic settings + environment loading
├── document_loader.py      # Multi-format document parsing
├── rag_engine.py           # ChromaDB vector store + retrieval
└── claude_client.py        # Anthropic API client with streaming
```

### Core Components

**RAG Engine (rag_engine.py)**
- ChromaDB for vector storage with persistence
- Sentence-transformers for embedding generation
- Chunking with overlap for context preservation
- Relevance scoring on retrieved documents

```python
class RAGEngine:
    def __init__(self, cfg: Config):
        self.client = chromadb.PersistentClient(path=cfg.chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
        )
        self.chunk_size = cfg.chunk_size
        self.chunk_overlap = cfg.chunk_overlap

    def ingest(self, pages: List[Page]) -> int:
        chunks = self._chunk_pages(pages)
        self.collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks]
        )
        return len(chunks)

    def query(self, question: str, top_k: int = 5) -> List[RetrievedChunk]:
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return self._format_results(results)
```

**Document Loader (document_loader.py)**
- PDF parsing with PyPDF2
- Word document extraction with python-docx
- Markdown/text file handling
- Page-level metadata preservation

```python
def load_from_bytes(content: bytes, filename: str) -> List[Page]:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _load_pdf(BytesIO(content), filename)
    elif ext == ".docx":
        return _load_docx(BytesIO(content), filename)
    elif ext in (".txt", ".md", ".markdown"):
        return _load_text(content.decode("utf-8"), filename)
    else:
        raise ValueError(f"Unsupported format: {ext}")
```

**Claude Client (claude_client.py)**
- Anthropic SDK integration
- Streaming response generation
- Conversation session management
- Token usage tracking and cost estimation

```python
class ClaudeClient:
    def __init__(self, cfg: Config):
        self.client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self.model = cfg.claude_model
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature

    def stream(self, session: ConversationSession, query: str, context: str) -> Generator[str, None, None]:
        system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context from documents:
{context}

Answer the question based on this context. If the answer isn't in the context, say so."""

        messages = session.get_messages() + [{"role": "user", "content": query}]

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text

        session.add_message("user", query)
        session.add_message("assistant", stream.get_final_message().content[0].text)
```

**Streamlit Interface (app.py)**
- Professional dark theme with custom CSS
- Two-column layout (chat + sources)
- Document upload with progress
- Session management (multiple conversations)
- RAG parameter tuning (Top-K, temperature)

### User Interface Features

```python
def render_chat(cfg, rag, claude) -> None:
    session = _active_session(claude)

    # Message history
    for msg in session.history:
        with st.chat_message(msg.role, avatar="🧑" if msg.role == "user" else "🤖"):
            st.markdown(msg.content)

    # Chat input with streaming response
    if prompt := st.chat_input("Ask a question about your documents…"):
        chunks = rag.query(prompt, top_k=cfg.top_k_results)
        context = rag.build_context(chunks)

        col_chat, col_sources = st.columns([3, 2])

        with col_chat:
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                full_response = ""

                for token in claude.stream(session, prompt, context):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

        with col_sources:
            _render_sources(chunks)  # Show retrieved context with relevance scores
```

## Key Features

| Feature | Implementation |
|---------|----------------|
| Vector Store | ChromaDB with persistence |
| Embeddings | Sentence Transformers |
| LLM | Claude (claude-3-sonnet-20240229) |
| Documents | PDF, DOCX, TXT, MD |
| Streaming | Real-time token display |
| Sessions | Multiple conversation threads |
| Source Display | Relevance bars + excerpts |

## Configuration

```env
# Anthropic API
ANTHROPIC_API_KEY=sk-ant-your-key-here
CLAUDE_MODEL=claude-3-sonnet-20240229
MAX_TOKENS=4096
TEMPERATURE=0.3

# RAG Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5

# Storage
CHROMA_DB_PATH=./chroma_db
UPLOADS_DIR=./uploads
```

## Deployment

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Production
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Technical Stats

- **Lines of Code**: ~1,556
- **Python Version**: 3.10+
- **Dependencies**: streamlit, anthropic, chromadb, sentence-transformers, pypdf2, python-docx
- **UI Framework**: Streamlit with custom CSS

## RAG Pipeline Flow

```
1. Document Upload
   └── Parse (PDF/DOCX/TXT)
       └── Split into chunks (1000 tokens, 200 overlap)
           └── Generate embeddings (sentence-transformers)
               └── Store in ChromaDB

2. User Query
   └── Embed query
       └── Retrieve top-K similar chunks
           └── Build context string
               └── Send to Claude with context
                   └── Stream response to UI
```

## Cost Estimation

| Model | Input | Output | ~Cost/Query |
|-------|-------|--------|-------------|
| Claude 3 Sonnet | $3/M tokens | $15/M tokens | ~$0.005 |
| Claude 3 Haiku | $0.25/M tokens | $1.25/M tokens | ~$0.0004 |

Cost tracking is built into the session management:
```python
st.metric("Session cost (est.)", f"${session.total_usage.cost_usd:.4f}")
```

---

**Author**: JustDreameritis
**Repository**: [github.com/JustDreameritis/ai-chatbot-demo](https://github.com/JustDreameritis/ai-chatbot-demo)
