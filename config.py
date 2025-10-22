# config.py
"""Global configuration for the medical RAG pipeline.

This module centralizes environment-driven settings and ensures required
directories exist. All secrets/endpoints use explicit FILL_ME markers so they
are easy to locate.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for bare environments
    def load_dotenv() -> bool:  # type: ignore[override]
        return False


# Load .env if present (silent no-op otherwise).
load_dotenv()


ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
ARTEFACTS_DIR = ROOT_DIR / "artefacts"

# Ensure key directories exist so downstream scripts can assume their presence.
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai_compat")  # "openai_compat" | "ollama"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://YOUR-LLM-ENDPOINT")  # === FILL_ME: set your endpoint URL ===
LLM_API_KEY = os.getenv("LLM_API_KEY", "YOUR-API-KEY")  # === FILL_ME: set your API key or leave blank if none ===
LLM_API_KEY_HEADER = os.getenv("LLM_API_KEY_HEADER", "Authorization")  # e.g., "Authorization" or "X-API-Key"
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-7b-instruct")  # === FILL_ME: model id/name at your endpoint ===

RETRIEVAL_TOPK = int(os.getenv("RETRIEVAL_TOPK", "5"))
RETRIEVAL_BM25_ALPHA = float(os.getenv("RETRIEVAL_BM25_ALPHA", "0.7"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "60"))

CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", str(DATA_DIR / "chunks.jsonl")))
INDEX_PATH = Path(os.getenv("INDEX_PATH", str(ARTEFACTS_DIR / "index.faiss")))
RECORDS_PATH = Path(os.getenv("RECORDS_PATH", str(ARTEFACTS_DIR / "records.pkl")))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def missing_llm_config() -> bool:
    """Return True when the LLM endpoint placeholder is still in use."""

    return "YOUR-LLM-ENDPOINT" in (LLM_BASE_URL or "")


__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "ARTEFACTS_DIR",
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_API_KEY_HEADER",
    "MISTRAL_MODEL",
    "RETRIEVAL_TOPK",
    "RETRIEVAL_BM25_ALPHA",
    "TIMEOUT_SECONDS",
    "CHUNKS_PATH",
    "INDEX_PATH",
    "RECORDS_PATH",
    "EMBED_MODEL_NAME",
    "LOG_LEVEL",
    "missing_llm_config",
]

print("=" * 60)
print("DEBUG: Configuration loaded from .env")
print("=" * 60)
print(f"LLM_PROVIDER = '{LLM_PROVIDER}'")
print(f"LLM_BASE_URL = '{LLM_BASE_URL}'")
print(f"LLM_API_KEY = '{LLM_API_KEY[:20] if LLM_API_KEY and len(LLM_API_KEY) > 20 else 'NOT SET'}...'")
print(f"LLM_API_KEY_HEADER = '{LLM_API_KEY_HEADER}'")
print(f"MISTRAL_MODEL = '{MISTRAL_MODEL}'")
print(f"missing_llm_config() = {missing_llm_config()}")
print("=" * 60)