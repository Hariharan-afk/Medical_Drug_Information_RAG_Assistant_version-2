from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from data_ingest import load_records
from generator import GenerationResult
from rag import RAGPipeline


class DummySentenceTransformer:
    def __init__(self, name: str) -> None:  # pragma: no cover - trivial init
        self.name = name

    def encode(
        self,
        texts,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ):
        arr = np.arange(len(texts) * 4, dtype="float32").reshape(len(texts), 4) + 1
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / norms
        return arr


class DummyBM25:
    def __init__(self, corpus_tokens):  # pragma: no cover - trivial init
        self.size = len(corpus_tokens)

    def get_scores(self, query_tokens):
        return np.ones(self.size)


def write_sample_jsonl(path: Path) -> None:
    records = [
        {
            "id": "chunk-1",
            "text": "Ibuprofen oral tablets should be taken with water to reduce stomach upset.",
            "drug_name": "ibuprofen",
            "route": "oral",
            "section_id": "PROPER_USE",
            "section_title": "Proper Use",
            "source": {"url": "https://example.org/a", "publisher": "Example"},
        },
        {
            "id": "chunk-2",
            "text": "Intravenous ibuprofen may increase bleeding risk; monitor patients clinically.",
            "drug_name": "ibuprofen",
            "route": "intravenous",
            "section_id": "SIDE_EFFECTS",
            "section_title": "Side Effects",
            "source": {"url": "https://example.org/b", "publisher": "Example"},
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_load_records(tmp_path):
    sample_path = tmp_path / "chunks.jsonl"
    write_sample_jsonl(sample_path)
    records = load_records(sample_path)
    assert len(records) == 2
    assert records[0]["id"] == "chunk-1"


def test_index_and_retrieve(tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    pytest.importorskip("rank_bm25")

    import index_builder
    import retriever

    sample_path = tmp_path / "chunks.jsonl"
    write_sample_jsonl(sample_path)

    monkeypatch.setattr(index_builder, "SentenceTransformer", DummySentenceTransformer)
    monkeypatch.setattr(retriever, "SentenceTransformer", DummySentenceTransformer)
    monkeypatch.setattr(retriever, "BM25Okapi", DummyBM25)

    index_path = tmp_path / "index.faiss"
    records_path = tmp_path / "records.pkl"

    index_builder.ensure_index(
        rebuild=True,
        records_path=records_path,
        index_path=index_path,
        chunks_path=sample_path,
        model_name="dummy",
    )

    r = retriever.Retriever(index_path=index_path, records_path=records_path, model_name="dummy")
    result = r.search("What are IV ibuprofen risks?", top_k=1, route="intravenous")
    assert result["results"], "Expected at least one retrieval result"


def test_rag_pipeline_with_stubs(monkeypatch):
    class StubRetriever:
        def search(self, query, top_k=5, route=None, drug=None):
            return {
                "results": [
                    {
                        "id": "chunk-1",
                        "text": "Intravenous ibuprofen may increase bleeding risk.",
                        "score": 0.9,
                        "section_title": "Side Effects",
                        "route": "intravenous",
                        "drug_name": "ibuprofen",
                        "url": "https://example.org/b",
                    }
                ],
                "notice": None,
            }

    class StubGenerator:
        def __init__(self):
            self.primary = None

        def generate(self, system_prompt, user_prompt, temperature=0.2, max_tokens=512):
            return GenerationResult(
                answer="Bleeding risk may increase [1]",
                backend="stub",
                used_fallback=False,
            )

    pipeline = RAGPipeline(retriever=StubRetriever(), generator=StubGenerator())
    result = pipeline.answer("What are IV ibuprofen risks?", top_k=1, route="intravenous")
    assert "Bleeding risk" in result["answer"]
    assert result["sources"], "Sources should not be empty"
