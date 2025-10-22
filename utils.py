# utils.py
"""Utility helpers shared across the RAG pipeline."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence


def setup_logger(name: str = "rag") -> logging.Logger:
    """Return a module-level logger with a friendly formatter."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(logging.getLogger().level or logging.INFO)
    return logger


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file, with basic validation."""

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} did not contain a JSON object")
            yield obj


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace so prompts stay compact."""

    return " ".join(text.split()) if text else ""


def take(iterable: Iterable, n: int) -> list:
    """Collect up to ``n`` items from an iterable without exhausting all data."""

    out = []
    for idx, item in enumerate(iterable):
        out.append(item)
        if idx + 1 >= n:
            break
    return out


def dedupe_preserve(seq: Sequence[str]) -> list[str]:
    """Remove duplicates while keeping original order."""

    return list(OrderedDict.fromkeys(seq))


def softmax(scores: Sequence[float]) -> list[float]:
    """Compute a numerically stable softmax distribution."""

    if not scores:
        return []
    max_score = max(scores)
    exps = [math.exp(s - max_score) for s in scores]
    total = sum(exps)
    return [val / total for val in exps] if total else [0.0 for _ in exps]


@contextmanager
def timer() -> Iterator[float]:
    """Context manager yielding elapsed seconds on exit."""

    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def sample_seeded(seq: Sequence, k: int, seed: int = 1337) -> list:
    """Return ``k`` pseudo-random elements using a stable seed."""

    rnd = random.Random(seed)
    if k >= len(seq):
        return list(seq)
    return rnd.sample(list(seq), k)


__all__ = [
    "setup_logger",
    "read_jsonl",
    "normalize_whitespace",
    "take",
    "dedupe_preserve",
    "softmax",
    "timer",
    "sample_seeded",
]

