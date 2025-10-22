# # # retriever.py
# # """Hybrid retriever combining FAISS semantic search with optional BM25."""

# # from __future__ import annotations

# # import math
# # import pickle
# # import re
# # from dataclasses import dataclass
# # from pathlib import Path
# # from typing import Dict, List, Optional, Sequence
# # from os import PathLike

# # import faiss
# # import numpy as np
# # from rank_bm25 import BM25Okapi
# # from sentence_transformers import SentenceTransformer

# # from config import (
# #     EMBED_MODEL_NAME,
# #     INDEX_PATH,
# #     RECORDS_PATH,
# #     RETRIEVAL_BM25_ALPHA,
# # )
# # from utils import setup_logger


# # logger = setup_logger("retriever")


# # @dataclass
# # class RetrievedChunk:
# #     id: str
# #     text: str
# #     score: float
# #     url: Optional[str]
# #     section_title: Optional[str]
# #     section_id: Optional[str]
# #     route: Optional[str]
# #     drug_name: Optional[str]


# # def _minmax(scores: Sequence[float]) -> List[float]:
# #     if not scores:
# #         return []
# #     low, high = min(scores), max(scores)
# #     if math.isclose(high, low):
# #         return [1.0 for _ in scores]
# #     return [(s - low) / (high - low) for s in scores]


# # def _tokenize(text: str) -> List[str]:
# #     return [tok.lower() for tok in text.split()]


# # class Retriever:
# #     def __init__(
# #         self,
# #         index_path: Optional[str | bytes | PathLike[str]] = None,
# #         records_path: Optional[str | bytes | PathLike[str]] = None,
# #         model_name: str = EMBED_MODEL_NAME,
# #         bm25_alpha: float = RETRIEVAL_BM25_ALPHA,
# #     ) -> None:
# #         self.index_path = INDEX_PATH if index_path is None else Path(index_path)
# #         self.records_path = RECORDS_PATH if records_path is None else Path(records_path)
# #         self.bm25_alpha = bm25_alpha
# #         self.model = SentenceTransformer(model_name)
# #         self.index = faiss.read_index(str(self.index_path))

# #         with self.records_path.open("rb") as fh:
# #             self.records: List[dict] = pickle.load(fh)

# #         logger.info("Loaded %s records for retrieval", len(self.records))
# #         self._build_bm25()
# #         self._build_metadata()

# #     def _build_bm25(self) -> None:
# #         corpus_tokens = [_tokenize(rec["text"]) for rec in self.records]
# #         self.bm25 = BM25Okapi(corpus_tokens)

# #     def _build_metadata(self) -> None:
# #         self.drug_index: Dict[str, set[int]] = {}
# #         self.drug_aliases: Dict[str, str] = {}
# #         self.section_index: Dict[str, set[int]] = {}
# #         self.section_aliases: Dict[str, str] = {}

# #         for idx, rec in enumerate(self.records):
# #             drug = (rec.get("drug_name") or "").strip()
# #             if drug:
# #                 canonical = drug.lower()
# #                 self.drug_index.setdefault(canonical, set()).add(idx)

# #         for canonical in self.drug_index:
# #             variants = {canonical, canonical.replace("-", " "), canonical.replace(" ", "-")}
# #             for variant in variants:
# #                 norm = variant.strip()
# #                 if norm:
# #                     self.drug_aliases[norm] = canonical

# #         for idx, rec in enumerate(self.records):
# #             section_id = (rec.get("section_id") or "").strip()
# #             section_title = (rec.get("section_title") or "").strip()

# #             canonical = None
# #             if section_id:
# #                 canonical = section_id.lower()
# #             elif section_title:
# #                 canonical = section_title.lower()

# #             if not canonical:
# #                 continue

# #             self.section_index.setdefault(canonical, set()).add(idx)

# #             synonyms = {canonical}
# #             synonyms.add(canonical.replace("_", " "))
# #             synonyms.add(canonical.replace("_", "-"))
# #             if section_title:
# #                 title_norm = section_title.lower()
# #                 synonyms.add(title_norm)
# #                 stripped = re.sub(r"[^a-z0-9 ]+", " ", title_norm).strip()
# #                 if stripped:
# #                     synonyms.add(stripped)

# #             for syn in list(synonyms):
# #                 if not syn:
# #                     continue
# #                 syn = re.sub(r"\s+", " ", syn).strip()
# #                 if len(syn) < 3:
# #                     continue
# #                 self.section_aliases[syn] = canonical

# #     def _semantic_search(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
# #         vec = self.model.encode(
# #             [query],
# #             convert_to_numpy=True,
# #             normalize_embeddings=True,
# #         )[0]
# #         vec = vec.astype(np.float32)
# #         scores, indices = self.index.search(vec.reshape(1, -1), top_k)
# #         return scores[0], indices[0]

# #     def _canonical_drug(self, value: Optional[str]) -> Optional[str]:
# #         if not value:
# #             return None
# #         key = value.lower()
# #         if key in self.drug_aliases:
# #             return self.drug_aliases[key]
# #         if key in self.drug_index:
# #             return key
# #         key_spaced = key.replace("-", " ")
# #         return self.drug_aliases.get(key_spaced)

# #     def _canonical_section(self, value: Optional[str]) -> Optional[str]:
# #         if not value:
# #             return None
# #         key = value.lower()
# #         if key in self.section_aliases:
# #             return self.section_aliases[key]
# #         return self.section_aliases.get(key.replace("-", " "))

# #     def _detect_query_filters(self, query: str) -> Dict[str, Optional[str]]:
# #         q = query.lower()
# #         detected_drug = None
# #         detected_section = None

# #         for alias, canonical in sorted(self.drug_aliases.items(), key=lambda x: -len(x[0])):
# #             if alias and alias in q:
# #                 detected_drug = canonical
# #                 break

# #         for alias, canonical in sorted(self.section_aliases.items(), key=lambda x: -len(x[0])):
# #             if alias and alias in q:
# #                 detected_section = canonical
# #                 break

# #         return {"drug": detected_drug, "section": detected_section}

# #     def search(
# #         self,
# #         query: str,
# #         top_k: int = 5,
# #         route: Optional[str] = None,
# #         drug: Optional[str] = None,
# #         auto_detect: bool = False,  # ADD THIS PARAMETER
# #     ) -> Dict[str, object]:
# #         if not query:
# #             return {"results": [], "notice": "Empty query"}

# #         detected = self._detect_query_filters(query) if auto_detect else {"drug": None, "section": None}
# #         top_k = max(1, top_k)
# #         candidate_k = max(50, top_k * 3)
# #         semantic_scores, semantic_indices = self._semantic_search(query, candidate_k)

# #         semantic_scores = list(map(float, semantic_scores))
# #         semantic_norm = _minmax(semantic_scores)

# #         query_tokens = _tokenize(query)
# #         bm25_scores_full = self.bm25.get_scores(query_tokens)
# #         bm25_scores = [float(bm25_scores_full[idx]) for idx in semantic_indices]
# #         bm25_norm = _minmax(bm25_scores)

# #         fused_scores: List[float] = []
# #         alpha = self.bm25_alpha
# #         for s, b in zip(semantic_norm, bm25_norm):
# #             fused_scores.append(alpha * s + (1 - alpha) * b)

# #         candidates: List[tuple[float, dict]] = []
# #         for idx, score in zip(semantic_indices, fused_scores):
# #             if idx == -1:
# #                 continue
# #             rec = self.records[idx]
# #             candidates.append((score, rec))

# #         # Apply filters (auto + explicit).
# #         route_lower = route.lower() if route else None
# #         explicit_drug = self._canonical_drug(drug)
# #         detected_drug = detected["drug"]
# #         filter_drug = self._canonical_drug(drug) or (detected["drug"] if auto_detect else None)
# #         detected_section = detected["section"]
# #         filter_section = detected["section"] if auto_detect else None

# #         matched: List[tuple[float, dict]] = []
# #         soft_matched: List[tuple[float, dict]] = []
# #         unmatched: List[tuple[float, dict]] = []

# #         for score, rec in candidates:
# #             rec_route = (rec.get("route") or "").lower() or None
# #             rec_drug = self._canonical_drug(rec.get("drug_name"))
# #             rec_section_id = self._canonical_section(rec.get("section_id"))
# #             rec_section_title = self._canonical_section(rec.get("section_title"))
# #             rec_section = rec_section_id or rec_section_title

# #             matches_route = not route_lower or route_lower == "any" or rec_route == route_lower
# #             matches_drug = not filter_drug or rec_drug == filter_drug
# #             matches_section = not filter_section or rec_section == filter_section

# #             # Perfect match
# #             if matches_route and matches_drug and matches_section:
# #                 matched.append((score, rec))
# #             # Partial match (boost slightly less)
# #             elif (matches_route or not route_lower) and (matches_drug or matches_section):
# #                 soft_matched.append((score * 0.95, rec))
# #             # No match (demote but don't eliminate)
# #             else:
# #                 unmatched.append((score * 0.7, rec))

# #         # Combine pools with priority
# #         selected_pool = matched + soft_matched + unmatched
# #         notice_parts: List[str] = []

# #         if route_lower and route_lower != "any" and not matched:
# #             route_hit = any(mr for _, _, mr, _, _ in unmatched)
# #             if not route_hit:
# #                 notice_parts.append("No exact route match found; showing closest results.")
# #         if filter_drug and not matched:
# #             drug_hit = any(md for _, _, _, md, _ in unmatched)
# #             if not drug_hit:
# #                 notice_parts.append("No exact drug match found; showing closest results.")
# #         if filter_section and not matched:
# #             section_hit = any(ms for _, _, _, _, ms in unmatched)
# #             if not section_hit:
# #                 notice_parts.append("No exact section match found; showing closest results.")

# #         selected_pool = sorted(selected_pool, key=lambda x: x[0], reverse=True)
# #         trimmed = selected_pool[:top_k]

# #         results: List[RetrievedChunk] = []
# #         for score, rec in trimmed:
# #             results.append(
# #                 RetrievedChunk(
# #                     id=rec["id"],
# #                     text=rec["text"],
# #                     score=float(score),
# #                     url=rec.get("url"),
# #                     section_title=rec.get("section_title"),
# #                     section_id=rec.get("section_id"),
# #                     route=rec.get("route"),
# #                     drug_name=rec.get("drug_name"),
# #                 )
# #             )

# #         return {
# #             "results": [r.__dict__ for r in results],
# #             "notice": " ".join(notice_parts) if notice_parts else None,
# #             "filters": {
# #                 "detected": detected,
# #                 "applied": {
# #                     "route": route_lower,
# #                     "drug": filter_drug,
# #                     "section": filter_section,
# #                 },
# #             },
# #         }


# # __all__ = ["Retriever", "RetrievedChunk"]


# """Hybrid retriever combining FAISS semantic search with optional BM25."""

# from __future__ import annotations

# import math
# import pickle
# import re
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Optional, Sequence
# from os import PathLike

# import faiss
# import numpy as np
# from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer

# from config import (
#     EMBED_MODEL_NAME,
#     INDEX_PATH,
#     RECORDS_PATH,
#     RETRIEVAL_BM25_ALPHA,
# )
# from utils import setup_logger


# logger = setup_logger("retriever")


# @dataclass
# class RetrievedChunk:
#     id: str
#     text: str
#     score: float
#     url: Optional[str]
#     section_title: Optional[str]
#     section_id: Optional[str]
#     route: Optional[str]
#     drug_name: Optional[str]


# def _minmax(scores: Sequence[float]) -> List[float]:
#     if not scores:
#         return []
#     low, high = min(scores), max(scores)
#     if math.isclose(high, low):
#         return [1.0 for _ in scores]
#     return [(s - low) / (high - low) for s in scores]


# def _tokenize(text: str) -> List[str]:
#     return [tok.lower() for tok in text.split()]


# class Retriever:
#     def __init__(
#         self,
#         index_path: Optional[str | bytes | PathLike[str]] = None,
#         records_path: Optional[str | bytes | PathLike[str]] = None,
#         model_name: str = EMBED_MODEL_NAME,
#         bm25_alpha: float = RETRIEVAL_BM25_ALPHA,
#     ) -> None:
#         self.index_path = INDEX_PATH if index_path is None else Path(index_path)
#         self.records_path = RECORDS_PATH if records_path is None else Path(records_path)
#         self.bm25_alpha = bm25_alpha
#         self.model = SentenceTransformer(model_name)
#         self.index = faiss.read_index(str(self.index_path))

#         with self.records_path.open("rb") as fh:
#             self.records: List[dict] = pickle.load(fh)

#         logger.info("Loaded %s records for retrieval", len(self.records))
#         self._build_bm25()
#         self._build_metadata()

#     def _build_bm25(self) -> None:
#         corpus_tokens = [_tokenize(rec["text"]) for rec in self.records]
#         self.bm25 = BM25Okapi(corpus_tokens)

#     def _build_metadata(self) -> None:
#         self.drug_index: Dict[str, set[int]] = {}
#         self.drug_aliases: Dict[str, str] = {}
#         self.section_index: Dict[str, set[int]] = {}
#         self.section_aliases: Dict[str, str] = {}

#         for idx, rec in enumerate(self.records):
#             drug = (rec.get("drug_name") or "").strip()
#             if drug:
#                 canonical = drug.lower()
#                 self.drug_index.setdefault(canonical, set()).add(idx)

#         for canonical in self.drug_index:
#             variants = {canonical, canonical.replace("-", " "), canonical.replace(" ", "-")}
#             for variant in variants:
#                 norm = variant.strip()
#                 if norm:
#                     self.drug_aliases[norm] = canonical

#         for idx, rec in enumerate(self.records):
#             section_id = (rec.get("section_id") or "").strip()
#             section_title = (rec.get("section_title") or "").strip()

#             canonical = None
#             if section_id:
#                 canonical = section_id.lower()
#             elif section_title:
#                 canonical = section_title.lower()

#             if not canonical:
#                 continue

#             self.section_index.setdefault(canonical, set()).add(idx)

#             synonyms = {canonical}
#             synonyms.add(canonical.replace("_", " "))
#             synonyms.add(canonical.replace("_", "-"))
#             if section_title:
#                 title_norm = section_title.lower()
#                 synonyms.add(title_norm)
#                 stripped = re.sub(r"[^a-z0-9 ]+", " ", title_norm).strip()
#                 if stripped:
#                     synonyms.add(stripped)

#             for syn in list(synonyms):
#                 if not syn:
#                     continue
#                 syn = re.sub(r"\s+", " ", syn).strip()
#                 if len(syn) < 3:
#                     continue
#                 self.section_aliases[syn] = canonical

#     def _semantic_search(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
#         vec = self.model.encode(
#             [query],
#             convert_to_numpy=True,
#             normalize_embeddings=True,
#         )[0]
#         vec = vec.astype(np.float32)
#         scores, indices = self.index.search(vec.reshape(1, -1), top_k)
#         return scores[0], indices[0]

#     def _canonical_drug(self, value: Optional[str]) -> Optional[str]:
#         if not value:
#             return None
#         key = value.lower()
#         if key in self.drug_aliases:
#             return self.drug_aliases[key]
#         if key in self.drug_index:
#             return key
#         key_spaced = key.replace("-", " ")
#         return self.drug_aliases.get(key_spaced)

#     def _canonical_section(self, value: Optional[str]) -> Optional[str]:
#         if not value:
#             return None
#         key = value.lower()
#         if key in self.section_aliases:
#             return self.section_aliases[key]
#         return self.section_aliases.get(key.replace("-", " "))

#     def search(
#         self,
#         query: str,
#         top_k: int = 5,
#         route: Optional[str] = None,
#         drug: Optional[str] = None,
#     ) -> Dict[str, object]:
#         if not query:
#             return {"results": [], "notice": "Empty query"}

#         # FIXED: Increased candidate pool from 50 to 200
#         top_k = max(1, top_k)
#         candidate_k = max(200, top_k * 20)
#         semantic_scores, semantic_indices = self._semantic_search(query, candidate_k)

#         semantic_scores = list(map(float, semantic_scores))
#         semantic_norm = _minmax(semantic_scores)

#         query_tokens = _tokenize(query)
#         bm25_scores_full = self.bm25.get_scores(query_tokens)
#         bm25_scores = [float(bm25_scores_full[idx]) for idx in semantic_indices]
#         bm25_norm = _minmax(bm25_scores)

#         fused_scores: List[float] = []
#         alpha = self.bm25_alpha
#         for s, b in zip(semantic_norm, bm25_norm):
#             fused_scores.append(alpha * s + (1 - alpha) * b)

#         candidates: List[tuple[float, dict]] = []
#         for idx, score in zip(semantic_indices, fused_scores):
#             if idx == -1:
#                 continue
#             rec = self.records[idx]
#             candidates.append((score, rec))

#         # FIXED: Softer filtering with graduated penalties
#         route_lower = route.lower() if route else None
#         filter_drug = self._canonical_drug(drug)  # FIXED: Removed auto-detection

#         scored_candidates: List[tuple[float, dict]] = []
#         notices: List[str] = []

#         for score, rec in candidates:
#             rec_route = (rec.get("route") or "").lower() or None
#             rec_drug = self._canonical_drug(rec.get("drug_name"))

#             # Calculate graduated multiplier
#             multiplier = 1.0

#             # Route filtering (minor penalty if mismatch)
#             if route_lower and route_lower != "any":
#                 if rec_route != route_lower:
#                     multiplier *= 0.95  # FIXED: 5% penalty instead of 30%

#             # Drug filtering (moderate penalty if explicitly set)
#             if filter_drug:
#                 if rec_drug != filter_drug:
#                     multiplier *= 0.85  # FIXED: 15% penalty instead of 30%

#             final_score = score * multiplier
#             scored_candidates.append((final_score, rec))

#         if not scored_candidates:
#             return {"results": [], "notice": "No candidates found."}

#         # Check if we have good matches
#         has_exact_match = any(score > 0.9 * max(s for s, _ in scored_candidates) 
#                               for score, _ in scored_candidates)

#         if route_lower and route_lower != "any" and not has_exact_match:
#             notices.append("No exact route match found; showing closest results.")
#         if filter_drug and not has_exact_match:
#             notices.append("No exact drug match found; showing closest results.")

#         scored_candidates = sorted(scored_candidates, key=lambda x: x[0], reverse=True)
#         trimmed = scored_candidates[:top_k]

#         results: List[RetrievedChunk] = []
#         for score, rec in trimmed:
#             results.append(
#                 RetrievedChunk(
#                     id=rec["id"],
#                     text=rec["text"],
#                     score=float(score),
#                     url=rec.get("url"),
#                     section_title=rec.get("section_title"),
#                     section_id=rec.get("section_id"),
#                     route=rec.get("route"),
#                     drug_name=rec.get("drug_name"),
#                 )
#             )

#         return {
#             "results": [r.__dict__ for r in results],
#             "notice": " ".join(notices) if notices else None,
#             "filters": {
#                 "applied": {
#                     "route": route_lower,
#                     "drug": filter_drug,
#                 },
#             },
#         }


# __all__ = ["Retriever", "RetrievedChunk"]

"""Hybrid retriever combining FAISS semantic search with optional BM25."""

from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from os import PathLike

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    EMBED_MODEL_NAME,
    INDEX_PATH,
    RECORDS_PATH,
    RETRIEVAL_BM25_ALPHA,
)
from utils import setup_logger


logger = setup_logger("retriever")


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    url: Optional[str]
    section_title: Optional[str]
    section_id: Optional[str]
    route: Optional[str]
    drug_name: Optional[str]


def _minmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if math.isclose(high, low):
        return [1.0 for _ in scores]
    return [(s - low) / (high - low) for s in scores]


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in text.split()]


class Retriever:
    def __init__(
        self,
        index_path: Optional[str | bytes | PathLike[str]] = None,
        records_path: Optional[str | bytes | PathLike[str]] = None,
        model_name: str = EMBED_MODEL_NAME,
        bm25_alpha: float = RETRIEVAL_BM25_ALPHA,
    ) -> None:
        self.index_path = INDEX_PATH if index_path is None else Path(index_path)
        self.records_path = RECORDS_PATH if records_path is None else Path(records_path)
        self.bm25_alpha = bm25_alpha
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(self.index_path))

        with self.records_path.open("rb") as fh:
            self.records: List[dict] = pickle.load(fh)

        logger.info("Loaded %s records for retrieval", len(self.records))
        self._build_bm25()
        self._build_metadata()

    def _build_bm25(self) -> None:
        corpus_tokens = [_tokenize(rec["text"]) for rec in self.records]
        self.bm25 = BM25Okapi(corpus_tokens)

    def _build_metadata(self) -> None:
        self.drug_index: Dict[str, set[int]] = {}
        self.drug_aliases: Dict[str, str] = {}
        self.section_index: Dict[str, set[int]] = {}
        self.section_aliases: Dict[str, str] = {}

        for idx, rec in enumerate(self.records):
            drug = (rec.get("drug_name") or "").strip()
            if drug:
                canonical = drug.lower()
                self.drug_index.setdefault(canonical, set()).add(idx)

        for canonical in self.drug_index:
            variants = {canonical, canonical.replace("-", " "), canonical.replace(" ", "-")}
            for variant in variants:
                norm = variant.strip()
                if norm:
                    self.drug_aliases[norm] = canonical

        for idx, rec in enumerate(self.records):
            section_id = (rec.get("section_id") or "").strip()
            section_title = (rec.get("section_title") or "").strip()

            canonical = None
            if section_id:
                canonical = section_id.lower()
            elif section_title:
                canonical = section_title.lower()

            if not canonical:
                continue

            self.section_index.setdefault(canonical, set()).add(idx)

            synonyms = {canonical}
            synonyms.add(canonical.replace("_", " "))
            synonyms.add(canonical.replace("_", "-"))
            if section_title:
                title_norm = section_title.lower()
                synonyms.add(title_norm)
                stripped = re.sub(r"[^a-z0-9 ]+", " ", title_norm).strip()
                if stripped:
                    synonyms.add(stripped)

            for syn in list(synonyms):
                if not syn:
                    continue
                syn = re.sub(r"\s+", " ", syn).strip()
                if len(syn) < 3:
                    continue
                self.section_aliases[syn] = canonical

    def _enhance_query(self, query: str, drug: Optional[str] = None, 
                       route: Optional[str] = None) -> str:
        """
        Enhance query with metadata to match embedding format.
        If drug/route filters provided, prepend them to boost relevance.
        """
        parts = []
        
        if drug:
            parts.append(drug)
        if route:
            parts.append(route)
        
        if parts:
            return " ".join(parts) + " " + query
        return query
    
    def _semantic_search(self, query: str, top_k: int, 
                        drug: Optional[str] = None, 
                        route: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
        # Enhance query to match embedding format
        enhanced_query = self._enhance_query(query, drug, route)
        
        vec = self.model.encode(
            [enhanced_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        vec = vec.astype(np.float32)
        scores, indices = self.index.search(vec.reshape(1, -1), top_k)
        return scores[0], indices[0]

    def _canonical_drug(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        key = value.lower()
        if key in self.drug_aliases:
            return self.drug_aliases[key]
        if key in self.drug_index:
            return key
        key_spaced = key.replace("-", " ")
        return self.drug_aliases.get(key_spaced)

    def _canonical_section(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        key = value.lower()
        if key in self.section_aliases:
            return self.section_aliases[key]
        return self.section_aliases.get(key.replace("-", " "))

    def search(
        self,
        query: str,
        top_k: int = 5,
        route: Optional[str] = None,
        drug: Optional[str] = None,
    ) -> Dict[str, object]:
        if not query:
            return {"results": [], "notice": "Empty query"}

        # FIXED: Increased candidate pool from 50 to 200
        top_k = max(1, top_k)
        candidate_k = max(200, top_k * 20)
        semantic_scores, semantic_indices = self._semantic_search(query, candidate_k)

        semantic_scores = list(map(float, semantic_scores))
        semantic_norm = _minmax(semantic_scores)

        query_tokens = _tokenize(query)
        bm25_scores_full = self.bm25.get_scores(query_tokens)
        bm25_scores = [float(bm25_scores_full[idx]) for idx in semantic_indices]
        bm25_norm = _minmax(bm25_scores)

        fused_scores: List[float] = []
        alpha = self.bm25_alpha
        for s, b in zip(semantic_norm, bm25_norm):
            fused_scores.append(alpha * s + (1 - alpha) * b)

        candidates: List[tuple[float, dict]] = []
        for idx, score in zip(semantic_indices, fused_scores):
            if idx == -1:
                continue
            rec = self.records[idx]
            candidates.append((score, rec))

        # FIXED: Softer filtering with graduated penalties
        route_lower = route.lower() if route else None
        filter_drug = self._canonical_drug(drug)  # FIXED: Removed auto-detection

        scored_candidates: List[tuple[float, dict]] = []
        notices: List[str] = []

        for score, rec in candidates:
            rec_route = (rec.get("route") or "").lower() or None
            rec_drug = self._canonical_drug(rec.get("drug_name"))

            # Calculate graduated multiplier
            multiplier = 1.0

            # Route filtering (minor penalty if mismatch)
            if route_lower and route_lower != "any":
                if rec_route != route_lower:
                    multiplier *= 0.95  # FIXED: 5% penalty instead of 30%

            # Drug filtering (moderate penalty if explicitly set)
            if filter_drug:
                if rec_drug != filter_drug:
                    multiplier *= 0.85  # FIXED: 15% penalty instead of 30%

            final_score = score * multiplier
            scored_candidates.append((final_score, rec))

        if not scored_candidates:
            return {"results": [], "notice": "No candidates found."}

        # Check if we have good matches
        has_exact_match = any(score > 0.9 * max(s for s, _ in scored_candidates) 
                              for score, _ in scored_candidates)

        if route_lower and route_lower != "any" and not has_exact_match:
            notices.append("No exact route match found; showing closest results.")
        if filter_drug and not has_exact_match:
            notices.append("No exact drug match found; showing closest results.")

        scored_candidates = sorted(scored_candidates, key=lambda x: x[0], reverse=True)
        trimmed = scored_candidates[:top_k]

        results: List[RetrievedChunk] = []
        for score, rec in trimmed:
            results.append(
                RetrievedChunk(
                    id=rec["id"],
                    text=rec["text"],
                    score=float(score),
                    url=rec.get("url"),
                    section_title=rec.get("section_title"),
                    section_id=rec.get("section_id"),
                    route=rec.get("route"),
                    drug_name=rec.get("drug_name"),
                )
            )

        return {
            "results": [r.__dict__ for r in results],
            "notice": " ".join(notices) if notices else None,
            "filters": {
                "applied": {
                    "route": route_lower,
                    "drug": filter_drug,
                },
            },
        }


__all__ = ["Retriever", "RetrievedChunk"]