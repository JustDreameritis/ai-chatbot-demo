"""
rag_engine.py — RAG pipeline: chunking, embedding, storage, retrieval

Uses:
  - Recursive text splitter with token-aware chunking
  - sentence-transformers (all-MiniLM-L6-v2) for embeddings
  - ChromaDB (local, no server) for vector storage
  - Cosine similarity for retrieval
"""

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config import Config
from document_loader import DocumentPage


@dataclass
class Chunk:
    """A text chunk ready for embedding and retrieval."""

    text: str
    chunk_id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk returned by similarity search, with its relevance score."""

    text: str
    metadata: dict
    score: float  # 0–1, higher = more similar
    chunk_id: str


class TextSplitter:
    """Recursive text splitter that respects sentence and paragraph boundaries.

    Tries to split on paragraphs → sentences → words, ensuring chunks
    don't exceed max_tokens and have the desired overlap.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """Initialize the splitter.

        Args:
            chunk_size: Target maximum tokens per chunk.
            chunk_overlap: Number of tokens to overlap between adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tokenizer = self._build_tokenizer()

    def _build_tokenizer(self):
        """Return a token counter function.

        Tries tiktoken (fast, accurate for Claude), falls back to word count.
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return lambda text: len(enc.encode(text))
        except Exception:
            return lambda text: len(text.split())

    def _token_count(self, text: str) -> int:
        return self._tokenizer(text)

    def split(self, text: str) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Input text to split.

        Returns:
            List of chunk strings.
        """
        chunks = self._recursive_split(text, self.SEPARATORS)
        return self._merge_with_overlap(chunks)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text, trying each separator in order.

        Args:
            text: Text to split.
            separators: Ordered list of separator strings to try.

        Returns:
            List of sub-strings at or below chunk_size.
        """
        if not separators:
            return [text]

        sep = separators[0]
        remaining = separators[1:]

        if sep:
            parts = text.split(sep)
        else:
            parts = list(text)

        result: List[str] = []
        current: List[str] = []
        current_len = 0

        for part in parts:
            part_len = self._token_count(part)
            if part_len > self.chunk_size and remaining:
                # This part alone is too big — recurse
                if current:
                    result.append(sep.join(current))
                    current, current_len = [], 0
                result.extend(self._recursive_split(part, remaining))
            elif current_len + part_len + (1 if current else 0) <= self.chunk_size:
                current.append(part)
                current_len += part_len + (1 if current else 0)
            else:
                if current:
                    result.append(sep.join(current))
                current = [part]
                current_len = part_len

        if current:
            result.append(sep.join(current))

        return [r for r in result if r.strip()]

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between adjacent chunks.

        Args:
            chunks: List of non-overlapping chunk strings.

        Returns:
            List of overlapping chunk strings.
        """
        if not chunks:
            return []
        if len(chunks) == 1:
            return chunks

        result: List[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
                continue

            # Prepend tail of previous chunk for context
            prev = chunks[i - 1]
            prev_words = prev.split()
            overlap_words = prev_words[-self.chunk_overlap:] if len(prev_words) > self.chunk_overlap else prev_words
            overlap_text = " ".join(overlap_words)

            merged = overlap_text + " " + chunk if overlap_text else chunk
            result.append(merged.strip())

        return result


class RAGEngine:
    """End-to-end RAG pipeline: ingest documents, store embeddings, retrieve context.

    Backed by ChromaDB (local persistent store) and sentence-transformers.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the RAG engine.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.splitter = TextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self._embedding_model = None
        self._chroma_client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Lazy initialisation (avoids slow imports at module load time)
    # ------------------------------------------------------------------

    def _get_embedding_model(self):
        """Lazy-load the sentence-transformer embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.config.embedding_model)
        return self._embedding_model

    def _get_collection(self):
        """Lazy-load the ChromaDB collection."""
        if self._collection is None:
            import chromadb

            self._chroma_client = chromadb.PersistentClient(
                path=self.config.chroma_db_path
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, pages: List[DocumentPage]) -> int:
        """Chunk, embed, and store document pages.

        Skips chunks that are already present (content-hash deduplication).

        Args:
            pages: List of DocumentPage objects from document_loader.

        Returns:
            Number of new chunks added.
        """
        collection = self._get_collection()
        model = self._get_embedding_model()

        all_chunks: List[Chunk] = []
        for page in pages:
            chunk_texts = self.splitter.split(page.text)
            for j, text in enumerate(chunk_texts):
                chunk_id = self._make_id(text)
                all_chunks.append(
                    Chunk(
                        text=text,
                        chunk_id=chunk_id,
                        metadata={
                            **page.metadata,
                            "chunk_index": j,
                        },
                    )
                )

        if not all_chunks:
            return 0

        # Filter already-stored chunks
        existing_ids = set(collection.get(ids=[c.chunk_id for c in all_chunks])["ids"])
        new_chunks = [c for c in all_chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            return 0

        texts = [c.text for c in new_chunks]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=[c.chunk_id for c in new_chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c.metadata for c in new_chunks],
        )

        return len(new_chunks)

    def query(self, query_text: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query_text: The user's question or search text.
            top_k: Number of results to return (defaults to config.top_k_results).

        Returns:
            List of RetrievedChunk objects ordered by relevance (best first).
        """
        k = top_k or self.config.top_k_results
        collection = self._get_collection()
        model = self._get_embedding_model()

        total = collection.count()
        if total == 0:
            return []

        k = min(k, total)
        query_embedding = model.encode([query_text], show_progress_bar=False).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[RetrievedChunk] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score 0–1
            score = max(0.0, 1.0 - dist / 2.0)
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    metadata=meta,
                    score=score,
                    chunk_id=self._make_id(doc),
                )
            )

        return chunks

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context block for Claude.

        Args:
            chunks: Retrieved chunks from query().

        Returns:
            Formatted context string to inject into the system prompt.
        """
        if not chunks:
            return ""

        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "?")
            score_pct = f"{chunk.score * 100:.0f}%"
            parts.append(
                f"[Source {i}: {source}, page {page}, relevance {score_pct}]\n"
                f"{chunk.text}"
            )

        return "\n\n---\n\n".join(parts)

    def document_count(self) -> int:
        """Return the number of unique source documents in the store.

        Returns:
            Count of distinct source filenames.
        """
        collection = self._get_collection()
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        sources = {m.get("source", "") for m in all_meta}
        return len(sources)

    def chunk_count(self) -> int:
        """Return total number of stored chunks.

        Returns:
            Integer count.
        """
        return self._get_collection().count()

    def list_documents(self) -> List[str]:
        """Return sorted list of source document names.

        Returns:
            List of filename strings.
        """
        collection = self._get_collection()
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        sources = sorted({m.get("source", "Unknown") for m in all_meta})
        return sources

    def delete_document(self, source_name: str) -> int:
        """Remove all chunks belonging to a document.

        Args:
            source_name: The 'source' metadata value used during ingestion.

        Returns:
            Number of chunks deleted.
        """
        collection = self._get_collection()
        results = collection.get(where={"source": source_name}, include=["metadatas"])
        ids = results["ids"]
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def clear_all(self) -> None:
        """Delete all documents and chunks from the store."""
        collection = self._get_collection()
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(text: str) -> str:
        """Deterministic chunk ID from content hash.

        Args:
            text: Chunk text.

        Returns:
            Hex digest string.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
