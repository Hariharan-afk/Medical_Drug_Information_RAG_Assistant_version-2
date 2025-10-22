# rag.py
"""High-level RAG orchestration for the medical assistant."""

from __future__ import annotations

from typing import Dict, List, Optional

from config import RETRIEVAL_TOPK
from generator import GenerationResult, GeneratorError, GeneratorRouter
from prompt_templates import SYSTEM_PROMPT, build_user_message
from utils import setup_logger

try:  # Allow import even when faiss is absent (e.g., docs or unit tests)
    from retriever import Retriever
except ModuleNotFoundError:  # pragma: no cover - raised when faiss is unavailable
    Retriever = None  # type: ignore[assignment]


logger = setup_logger("rag")

DISCLAIMER = (
    "This is not medical advice. For decisions about medicines, talk to a licensed clinician."
)


def _truncate(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rsplit(" ", 1)[0] + "..."


class RAGPipeline:
    def __init__(self, retriever: Optional["Retriever"] = None, generator: Optional[GeneratorRouter] = None) -> None:
        if retriever is None:
            if Retriever is None:
                raise RuntimeError(
                    "Retriever backend is unavailable. Install 'faiss-cpu' and 'rank-bm25' to enable retrieval."
                )
            retriever = Retriever()
        self.retriever = retriever
        self.generator = generator or GeneratorRouter()

    def _build_contexts(self, chunks: List[dict]) -> List[dict]:
        contexts = []
        seen_ids = set()
        for idx, chunk in enumerate(chunks, start=1):
            if chunk["id"] in seen_ids:
                continue
            seen_ids.add(chunk["id"])
            contexts.append(
                {
                    "text": _truncate(chunk["text"]),
                    "citation": f"[{idx}]",
                    "section": chunk.get("section_title"),
                    "route": chunk.get("route"),
                    "drug_name": chunk.get("drug_name"),
                }
            )
        return contexts

    def _ensure_disclaimer(self, answer: str) -> str:
        if DISCLAIMER.lower() in answer.lower():
            return answer
        return answer.strip() + "\n\n" + DISCLAIMER

    def answer(
        self,
        question: str,
        top_k: int = RETRIEVAL_TOPK,
        temperature: float = 0.2,
        route: Optional[str] = None,
        drug: Optional[str] = None,
    ) -> Dict[str, object]:
        retrieval = self.retriever.search(question, top_k=top_k, route=route, drug=drug)
        chunks = retrieval.get("results", [])
        if not chunks:
            fallback_answer = (
                "I could not find relevant information in the knowledge base. "
                "Consider consulting a clinician for guidance."
            )
            return {
                "answer": self._ensure_disclaimer(fallback_answer),
                "sources": [],
                "backend": None,
                "used_fallback": True,
                "retrieval_notice": retrieval.get("notice"),
                "retrieval_debug": [],
            }

        contexts = self._build_contexts(chunks)
        user_prompt = build_user_message(question, contexts)

        try:
            generation: GenerationResult = self.generator.generate(
                SYSTEM_PROMPT,
                user_prompt,
                temperature=temperature,
                max_tokens=512,
            )
            answer = self._ensure_disclaimer(generation.answer)
        except GeneratorError as exc:
            logger.error("All generators failed: %s", exc)
            fallback_answer = (
                "I was unable to generate an answer due to a model error. "
                "Please try again later or consult a clinician."
            )
            return {
                "answer": self._ensure_disclaimer(fallback_answer),
                "sources": [],
                "backend": None,
                "used_fallback": True,
                "retrieval_notice": retrieval.get("notice"),
                "retrieval_debug": [],
            }

        sources: List[dict] = []
        seen_sources = set()
        for chunk in chunks:
            key = (
                chunk.get("url") or chunk.get("id"),
                (chunk.get("drug_name") or "").lower(),
                (chunk.get("route") or "").lower(),
            )
            if key in seen_sources:
                continue
            seen_sources.add(key)
            sources.append(
                {
                    "id": chunk["id"],
                    "url": chunk.get("url"),
                    "section": chunk.get("section_title"),
                    "route": chunk.get("route"),
                    "drug_name": chunk.get("drug_name"),
                }
            )

        retrieval_debug = [
            {
                "id": chunk["id"],
                "score": round(float(chunk["score"]), 4),
                "preview": _truncate(chunk["text"], 280),
                "section": chunk.get("section_title"),
                "url": chunk.get("url"),
            }
            for chunk in chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "backend": generation.backend,
            "used_fallback": generation.used_fallback,
            "retrieval_notice": retrieval.get("notice"),
            "retrieval_debug": retrieval_debug,
        }


__all__ = ["RAGPipeline"]
