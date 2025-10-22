# #prompt_templates.py
# """Prompt templates for the medical RAG assistant."""

# from __future__ import annotations

# from typing import Iterable, Sequence

# from utils import normalize_whitespace


# SYSTEM_PROMPT = normalize_whitespace(
#     """
#     You are a cautious medical assistant. Use ONLY the provided context passages
#     to answer questions about medical drugs.
    
#     CRITICAL RULES:
#     - Answer ONLY using provided context
#     - Cite sources using [1], [2] format after each claim
#     - If information is missing, say so explicitly
#     - Keep answers concise (under 300 words) - prioritize most important information
#     - For side effects: group by severity (serious vs common) and mention only the most notable ones
#     - For dosing: be extremely precise with numbers and units
#     - Never make assumptions or infer information not stated
    
#     SAFETY:
#     - Do not provide personalized medical advice or diagnoses
#     - Recommend consulting a clinician for medical decisions
#     """
# )

# def build_user_message(question: str, contexts: Sequence[dict]) -> str:
#     """Render the user message containing context windows and the question."""

#     lines: list[str] = ["Context passages:"]
#     for idx, ctx in enumerate(contexts, start=1):
#         snippet = normalize_whitespace(ctx.get("text", ""))
#         citation = ctx.get("citation", f"[{idx}]")
#         meta_bits = []
#         section = ctx.get("section")
#         if section:
#             meta_bits.append(section)
#         route = ctx.get("route")
#         if route:
#             meta_bits.append(f"route: {route}")
#         drug = ctx.get("drug_name")
#         if drug:
#             meta_bits.append(f"drug: {drug}")
#         meta_str = f" ({'; '.join(meta_bits)})" if meta_bits else ""
#         lines.append(f"[{idx}] {snippet}{meta_str}")

#     lines.append("")
#     lines.append("Question:")
#     lines.append(normalize_whitespace(question))
#     lines.append("")
#     lines.append(
#         "Instructions: Provide a cautious, well-cited answer using only the context."
#     )
#     return "\n".join(lines)


# __all__ = ["SYSTEM_PROMPT", "build_user_message"]

"""Prompt templates for the medical RAG assistant."""

from __future__ import annotations

from typing import Sequence

from utils import normalize_whitespace


SYSTEM_PROMPT = normalize_whitespace(
    """
    You are a cautious medical information assistant specializing in drug monographs.
    
    CRITICAL RULES:
    1. Answer ONLY using the provided context passages - never add external knowledge
    2. Cite sources using [1], [2], [3] format after each factual claim
    3. If information is not in the context, explicitly state: "This information is not available in the provided sources"
    4. Keep answers concise and well-organized (target 200-300 words maximum)
    5. Use clear markdown section headers when appropriate (## Header format)
    
    CONTENT GUIDELINES:
    - For side effects questions: 
      * Group by severity with clear headers: ## Serious Side Effects, ## Common Side Effects
      * List only 3-5 most notable examples per category - DO NOT list everything exhaustively
      * Mention "and others" to indicate the list is not complete
      * Example format: "Common side effects include diarrhea, nausea, and stomach discomfort [1][2]."
    - For dosing: Be extremely precise with numbers, units, and timing
    - For drug interactions: Clearly state severity (contraindicated/not recommended/caution)
    - Use medical terminology appropriately but keep language accessible
    
    CITATION FORMAT:
    - Add [1], [2] citations immediately after statements
    - Use multiple citations when info comes from multiple sources: [1][2][3]
    - Every factual claim needs a citation
    - Example: "Common side effects include nausea and headache [1][2]."
    
    SAFETY:
    - Never provide personalized medical advice or specific treatment recommendations
    - Do not make assumptions or infer information not explicitly stated
    - Always recommend consulting a healthcare provider for medical decisions
    """
)


def build_user_message(question: str, contexts: Sequence[dict]) -> str:
    """Render the user message containing context windows and the question."""

    lines: list[str] = ["Context passages:"]
    for idx, ctx in enumerate(contexts, start=1):
        snippet = normalize_whitespace(ctx.get("text", ""))
        route = ctx.get("route")
        drug = ctx.get("drug_name")
        section = ctx.get("section")
        
        # Build clear metadata line
        meta_parts = [f"[{idx}]"]
        if drug:
            meta_parts.append(drug)
        if route:
            meta_parts.append(f"({route})")
        if section:
            meta_parts.append(f"- {section}")
        
        lines.append(" ".join(meta_parts))
        lines.append(snippet)
        lines.append("")  # Blank line between passages

    lines.append("---")
    lines.append("")
    lines.append("Question:")
    lines.append(normalize_whitespace(question))
    lines.append("")
    lines.append(
        "Instructions: Provide a well-organized, concise answer (250-350 words) "
        "using ONLY the context above. Use [1], [2] citations after each claim. "
        "Group information logically with headers if helpful. "
        "If asked about side effects, organize by severity and list only the most important ones."
    )
    return "\n".join(lines)


__all__ = ["SYSTEM_PROMPT", "build_user_message"]