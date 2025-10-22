# data_ingest.py
"""Load and normalize chunk records for the RAG pipeline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, List, Optional, TypedDict

from config import CHUNKS_PATH
from utils import setup_logger


logger = setup_logger("data_ingest")


class NormalizedRecord(TypedDict, total=False):
    id: str
    text: str
    drug_name: Optional[str]
    route: Optional[str]
    section_id: Optional[str]
    section_title: Optional[str]
    url: Optional[str]
    publisher: Optional[str]


def normalize_record(raw: dict) -> Optional[NormalizedRecord]:
    """Return a canonical record or ``None`` if mandatory fields are missing."""

    text = (raw.get("text") or "").strip()
    if not text:
        return None

    rid = raw.get("id") or raw.get("chunk_id") or raw.get("uuid")
    if not rid:
        return None

    source = raw.get("source") or {}
    citations = raw.get("citations") or []
    if not source and citations and isinstance(citations, list):
        # Fall back to first citation if present.
        source = citations[0] if isinstance(citations[0], dict) else {}

    return NormalizedRecord(
        id=str(rid),
        text=text,
        drug_name=(raw.get("drug_name") or raw.get("drug") or "").strip() or None,
        route=(raw.get("route") or raw.get("drug_route") or "").strip() or None,
        section_id=(raw.get("section_id") or raw.get("section") or "").strip() or None,
        section_title=(raw.get("section_title") or raw.get("section") or "").strip() or None,
        url=(source.get("url") or "").strip() or None,
        publisher=(source.get("publisher") or "").strip() or None,
    )


def load_records(path: Path) -> List[NormalizedRecord]:
    """Load and normalize JSONL records, skipping malformed entries."""

    records: List[NormalizedRecord] = []
    bad = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                bad += 1
                logger.warning("Skipping line %s: %s", line_no, exc)
                continue
            if not isinstance(raw, dict):
                bad += 1
                logger.warning("Skipping line %s: expected JSON object", line_no)
                continue
            norm = normalize_record(raw)
            if norm is None:
                bad += 1
                logger.debug("Skipping line %s: missing required fields", line_no)
                continue
            records.append(norm)

    if bad:
        logger.warning("Skipped %s malformed records", bad)

    if not records:
        raise RuntimeError(f"No usable records were loaded from {path}")

    lengths = [len(r["text"]) for r in records]
    logger.info(
        "Loaded %s records | mean chars %.1f | median chars %.1f",
        len(records),
        statistics.fmean(lengths),
        statistics.median(lengths),
    )
    return records


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Load chunk records and print stats")
    parser.add_argument("--path", type=Path, default=CHUNKS_PATH, help="Path to chunks JSONL")
    args = parser.parse_args(argv)

    records = load_records(args.path)
    logger.info("First record preview: %s", records[0].get("section_title"))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
