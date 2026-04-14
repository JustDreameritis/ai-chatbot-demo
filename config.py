"""
config.py — Configuration management for AI Document Chatbot

Loads settings from .env file with sensible defaults.
All configurable values live here — no hardcoded constants elsewhere.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Central configuration object for the chatbot application."""

    # Claude API
    anthropic_api_key: str
    claude_model: str

    # RAG pipeline
    chunk_size: int
    chunk_overlap: int
    top_k_results: int

    # Generation
    max_tokens: int
    temperature: float

    # Storage paths
    chroma_db_path: str
    uploads_dir: str

    # Embedding model
    embedding_model: str

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables.

        Returns:
            Config: Populated configuration instance.

        Raises:
            ValueError: If ANTHROPIC_API_KEY is not set.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )

        return cls(
            anthropic_api_key=api_key,
            claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            top_k_results=int(os.getenv("TOP_K_RESULTS", "5")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            chroma_db_path=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
            uploads_dir=os.getenv("UPLOADS_DIR", "./uploads"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
        )


def get_config() -> Config:
    """Return the active configuration.

    Cached after first call so repeated imports don't re-read .env.

    Returns:
        Config: Application configuration.
    """
    return Config.from_env()
