# # index_builder.py
# """Build or load the FAISS index for chunk retrieval."""

# from __future__ import annotations

# import argparse
# import pickle
# from pathlib import Path
# from typing import Iterable, Optional

# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer

# from config import (
#     ARTEFACTS_DIR,
#     CHUNKS_PATH,
#     EMBED_MODEL_NAME,
#     INDEX_PATH,
#     RECORDS_PATH,
# )
# from data_ingest import load_records
# from utils import setup_logger


# logger = setup_logger("index_builder")


# def build_index(
#     records_path: Path,
#     index_path: Path,
#     chunks_path: Path,
#     model_name: str,
# ) -> None:
#     records = load_records(chunks_path)
#     logger.info("Encoding %s records with %s", len(records), model_name)

#     model = SentenceTransformer(model_name)
#     texts = [rec["text"] for rec in records]
#     embeddings = model.encode(
#         texts,
#         batch_size=64,
#         show_progress_bar=True,
#         convert_to_numpy=True,
#         normalize_embeddings=True,
#     )

#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(embeddings.astype(np.float32))

#     index_path.parent.mkdir(parents=True, exist_ok=True)
#     records_path.parent.mkdir(parents=True, exist_ok=True)

#     faiss.write_index(index, str(index_path))
#     with records_path.open("wb") as fh:
#         pickle.dump(records, fh)

#     logger.info("Wrote index to %s and records to %s", index_path, records_path)


# def ensure_index(
#     rebuild: bool,
#     records_path: Path,
#     index_path: Path,
#     chunks_path: Path,
#     model_name: str,
# ) -> None:
#     if rebuild or not index_path.exists() or not records_path.exists():
#         logger.info("(Re)building index")
#         build_index(records_path, index_path, chunks_path, model_name)
#     else:
#         logger.info("Index already present at %s", index_path)


# def main(argv: Optional[Iterable[str]] = None) -> int:
#     parser = argparse.ArgumentParser(description="Build FAISS index for drug chunks")
#     parser.add_argument("--chunks-path", type=Path, default=CHUNKS_PATH)
#     parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
#     parser.add_argument("--records-path", type=Path, default=RECORDS_PATH)
#     parser.add_argument("--model", default=EMBED_MODEL_NAME)
#     parser.add_argument("--rebuild", action="store_true")
#     args = parser.parse_args(argv)

#     ensure_index(
#         rebuild=args.rebuild,
#         records_path=args.records_path,
#         index_path=args.index_path,
#         chunks_path=args.chunks_path,
#         model_name=args.model,
#     )
#     return 0


# if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
#     raise SystemExit(main())

"""Build or load the FAISS index for chunk retrieval."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    ARTEFACTS_DIR,
    CHUNKS_PATH,
    EMBED_MODEL_NAME,
    INDEX_PATH,
    RECORDS_PATH,
)
from data_ingest import load_records
from utils import setup_logger


logger = setup_logger("index_builder")


def build_embedding_text(rec: dict) -> str:
    """
    Build metadata-enhanced text for embedding.
    Format: {drug} {route} {section}: {text}
    
    This ensures the embedding model learns drug-specific semantic representations.
    """
    parts = []
    
    # Add drug name
    if rec.get("drug_name"):
        parts.append(rec["drug_name"])
    
    # Add route
    if rec.get("route"):
        parts.append(rec["route"])
    
    # Add section
    section = rec.get("section_title") or rec.get("section_id")
    if section:
        parts.append(section)
    
    # Combine metadata with text
    if parts:
        prefix = " ".join(parts) + ": "
    else:
        prefix = ""
    
    return prefix + rec["text"]


def build_index(
    records_path: Path,
    index_path: Path,
    chunks_path: Path,
    model_name: str,
) -> None:
    records = load_records(chunks_path)
    logger.info("Encoding %s records with %s", len(records), model_name)

    model = SentenceTransformer(model_name)
    
    # FIXED: Use metadata-enhanced text for embedding
    texts = [build_embedding_text(rec) for rec in records]
    
    # Debug: Show sample embedding text
    logger.info("Sample embedding text (first 3 records):")
    for i, text in enumerate(texts[:3], 1):
        preview = text[:150] + "..." if len(text) > 150 else text
        logger.info(f"  [{i}] {preview}")
    
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    with records_path.open("wb") as fh:
        pickle.dump(records, fh)

    logger.info("Wrote index to %s and records to %s", index_path, records_path)


def ensure_index(
    rebuild: bool,
    records_path: Path,
    index_path: Path,
    chunks_path: Path,
    model_name: str,
) -> None:
    if rebuild or not index_path.exists() or not records_path.exists():
        logger.info("(Re)building index")
        build_index(records_path, index_path, chunks_path, model_name)
    else:
        logger.info("Index already present at %s", index_path)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build FAISS index for drug chunks")
    parser.add_argument("--chunks-path", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    parser.add_argument("--records-path", type=Path, default=RECORDS_PATH)
    parser.add_argument("--model", default=EMBED_MODEL_NAME)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    ensure_index(
        rebuild=args.rebuild,
        records_path=args.records_path,
        index_path=args.index_path,
        chunks_path=args.chunks_path,
        model_name=args.model,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())