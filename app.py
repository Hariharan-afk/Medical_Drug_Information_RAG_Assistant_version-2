# # app.py
# """Gradio application entrypoint for the medical RAG assistant."""

# from __future__ import annotations

# import json
# import tempfile
# from typing import Optional, Tuple

# import gradio as gr
# import gradio_client.utils as gr_utils

# from config import (
#     CHUNKS_PATH,
#     EMBED_MODEL_NAME,
#     INDEX_PATH,
#     RECORDS_PATH,
#     missing_llm_config,
# )
# from index_builder import ensure_index
# from rag import DISCLAIMER, RAGPipeline


# _original_get_type = gr_utils.get_type
# def _safe_get_type(schema):
#     if isinstance(schema, bool):
#         return "boolean"
#     return _original_get_type(schema)


# _original_json_schema = gr_utils._json_schema_to_python_type
# def _safe_json_schema(schema, defs=None):
#     if isinstance(schema, bool):
#         return "boolean"
#     return _original_json_schema(schema, defs)


# gr_utils.get_type = _safe_get_type
# gr_utils._json_schema_to_python_type = _safe_json_schema


# def bootstrap_pipeline() -> Tuple[RAGPipeline, Optional[str]]:
#     needs_rebuild = not INDEX_PATH.exists() or not RECORDS_PATH.exists()
#     if needs_rebuild:
#         print("Building FAISS index (first run may take several minutes)...")
#     ensure_index(needs_rebuild, RECORDS_PATH, INDEX_PATH, CHUNKS_PATH, EMBED_MODEL_NAME)
#     pipeline = RAGPipeline()
#     banner = None
#     if pipeline.generator.primary is None:
#         banner = "Using local FLAN-T5 fallback (CPU). Configure LLM_BASE_URL to enable Mistral API."
#     elif missing_llm_config():
#         banner = "LLM_BASE_URL looks like a placeholder; verify your environment settings."
#     return pipeline, banner


# pipeline, startup_banner = bootstrap_pipeline()


# def format_sources(sources: list[dict]) -> str:
#     if not sources:
#         return "No sources found."

#     unique: list[dict] = []
#     seen = set()
#     for src in sources:
#         key = (
#             src.get("url") or src.get("id"),
#             (src.get("drug_name") or "").lower(),
#             (src.get("route") or "").lower(),
#         )
#         if key in seen:
#             continue
#         seen.add(key)
#         unique.append(src)

#     lines = ["### Sources"]
#     for idx, src in enumerate(unique, start=1):
#         url = src.get("url") or "(no URL provided)"
#         drug = src.get("drug_name") or "Unknown drug"
#         route = src.get("route")
#         route_str = f" - route: {route}" if route else ""
#         lines.append(f"[{idx}] {drug}{route_str}\n{url}")
#     return "\n\n".join(lines)


# def format_debug(debug_rows: list[dict]) -> str:
#     if not debug_rows:
#         return "No retrieval debug data."
#     lines = ["### Retrieval Debug"]
#     for row in debug_rows:
#         lines.append(
#             f"- score={row['score']:.4f} | section={row.get('section') or 'n/a'} | url={row.get('url') or 'n/a'}\n"
#             f"  preview: {row['preview']}"
#         )
#     return "\n".join(lines)


# def _create_download_file(payload: dict) -> str:
#     tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8")
#     json.dump(payload, tmp, indent=2)
#     tmp.flush()
#     tmp.close()
#     return tmp.name


# def generate_answer(question: str, route: str, top_k: int, temperature: float, show_debug: bool):
#     if not question.strip():
#         warning = "Please enter a question about ibuprofen." + "\n\n" + DISCLAIMER
#         return (
#             warning,
#             "",
#             gr.update(value="", visible=show_debug),
#             startup_banner or "",
#             gr.update(value=None, visible=False),
#         )

#     route_filter = None if route == "any" else route
#     result = pipeline.answer(
#         question,
#         top_k=int(top_k),
#         temperature=temperature,
#         route=route_filter,
#         drug="ibuprofen",
#     )

#     answer_md = result["answer"]
#     sources_md = format_sources(result["sources"])
#     debug_md = format_debug(result["retrieval_debug"]) if show_debug else ""
#     debug_component = gr.update(value=debug_md, visible=show_debug)

#     notices = []
#     if startup_banner:
#         notices.append(startup_banner)
#     if result.get("retrieval_notice"):
#         notices.append(result["retrieval_notice"])
#     if result.get("backend"):
#         backend_label = result["backend"]
#         if result.get("used_fallback"):
#             backend_label += " (fallback)"
#         notices.append(f"Answer generated with: {backend_label}")
#     notice_md = "\n".join(notices)

#     payload_path = _create_download_file(result)
#     download = gr.update(value=payload_path, visible=True)

#     return answer_md, sources_md, debug_component, notice_md, download


# with gr.Blocks(title="Drug Monograph RAG (Ibuprofen) - Mistral API") as demo:
#     if startup_banner:
#         gr.Markdown(f"**{startup_banner}**")

#     gr.Markdown(
#         "Ask evidence-grounded questions about ibuprofen. Responses cite the underlying monograph passages."
#     )

#     with gr.Row():
#         question = gr.Textbox(
#             label="Ask a medical question about ibuprofen...",
#             placeholder="e.g., What are the IV ibuprofen risks?",
#             lines=3,
#         )

#     with gr.Row():
#         route = gr.Dropdown(
#             choices=["any", "oral", "intravenous"],
#             value="any",
#             label="Route filter",
#         )
#         topk = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-k passages")
#         temp = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Temperature")
#         show_debug = gr.Checkbox(value=False, label="Show retrieval debug")

#     run_button = gr.Button("Search & Answer", variant="primary")

#     answer_md = gr.Markdown(label="Answer")
#     sources_md = gr.Markdown(label="Sources")
#     debug_md = gr.Markdown(visible=False)
#     notice_md = gr.Markdown()
#     download_button = gr.DownloadButton(label="Export JSON", visible=False)

#     run_button.click(
#         generate_answer,
#         inputs=[question, route, topk, temp, show_debug],
#         outputs=[answer_md, sources_md, debug_md, notice_md, download_button],
#     )


# if __name__ == "__main__":
#     demo.queue(max_size=8)
#     from fastapi import FastAPI
#     import uvicorn

#     fastapi_app = FastAPI()
#     fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/")
#     print("Gradio app running on http://127.0.0.1:7861")
#     uvicorn.run(fastapi_app, host="127.0.0.1", port=7861)

# app.py
"""Modern Gradio application with enhanced UX for medical RAG assistant."""

from __future__ import annotations

import json
import tempfile
from typing import Optional, Tuple

import gradio as gr
import gradio_client.utils as gr_utils

from config import (
    CHUNKS_PATH,
    EMBED_MODEL_NAME,
    INDEX_PATH,
    RECORDS_PATH,
    missing_llm_config,
)
from index_builder import ensure_index
from rag import DISCLAIMER, RAGPipeline


_original_get_type = gr_utils.get_type
def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return _original_get_type(schema)


_original_json_schema = gr_utils._json_schema_to_python_type
def _safe_json_schema(schema, defs=None):
    if isinstance(schema, bool):
        return "boolean"
    return _original_json_schema(schema, defs)


gr_utils.get_type = _safe_get_type
gr_utils._json_schema_to_python_type = _safe_json_schema


def bootstrap_pipeline() -> Tuple[RAGPipeline, Optional[str]]:
    needs_rebuild = not INDEX_PATH.exists() or not RECORDS_PATH.exists()
    if needs_rebuild:
        print("Building FAISS index (first run may take several minutes)...")
    ensure_index(needs_rebuild, RECORDS_PATH, INDEX_PATH, CHUNKS_PATH, EMBED_MODEL_NAME)
    pipeline = RAGPipeline()
    banner = None
    if pipeline.generator.primary is None:
        banner = "⚠️ Using local FLAN-T5 fallback (CPU). Configure LLM_BASE_URL in .env for faster responses."
    elif missing_llm_config():
        banner = "⚠️ LLM configuration may be incomplete. Check your .env file."
    return pipeline, banner


pipeline, startup_banner = bootstrap_pipeline()


# Example queries for quick testing
EXAMPLE_QUERIES = [
    "What are the side effects of azithromycin?",
    "What is the correct dosage for oral azithromycin in adults?",
    "Can azithromycin be used during pregnancy?",
    "What drugs interact with azithromycin?",
    "What are the differences between oral and IV azithromycin?",
]


def format_sources_enhanced(sources: list[dict]) -> str:
    """Format sources with better visual presentation."""
    if not sources:
        return "### 📚 Sources\n\nNo sources found."

    unique: list[dict] = []
    seen = set()
    for src in sources:
        key = (
            src.get("url") or src.get("id"),
            (src.get("drug_name") or "").lower(),
            (src.get("route") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(src)

    lines = [f"### 📚 Sources ({len(unique)} documents)\n"]
    for idx, src in enumerate(unique, start=1):
        url = src.get("url") or "(no URL provided)"
        drug = src.get("drug_name") or "Unknown drug"
        route = src.get("route")
        section = src.get("section")
        
        # Format with emoji and better structure
        drug_route = f"**{drug}**"
        if route:
            drug_route += f" • *{route} route*"
        if section:
            drug_route += f" • {section}"
        
        lines.append(f"**[{idx}]** {drug_route}")
        lines.append(f"🔗 [{url}]({url})")
        lines.append("")
    
    return "\n".join(lines)


def format_debug_enhanced(debug_rows: list[dict]) -> str:
    """Format debug info with scores and previews."""
    if not debug_rows:
        return "No retrieval debug data available."
    
    lines = ["### 🔍 Retrieval Debug Information\n"]
    lines.append("| Rank | Score | Drug | Section | Preview |")
    lines.append("|------|-------|------|---------|---------|")
    
    for rank, row in enumerate(debug_rows, 1):
        score = row['score']
        drug = row.get('drug_name', 'N/A')
        section = row.get('section', 'N/A')
        preview = row['preview'][:80] + "..." if len(row['preview']) > 80 else row['preview']
        
        # Color code scores
        score_emoji = "🟢" if score > 0.85 else "🟡" if score > 0.75 else "🔴"
        
        lines.append(f"| {rank} | {score_emoji} {score:.3f} | {drug} | {section} | {preview} |")
    
    return "\n".join(lines)


def _create_download_file(payload: dict) -> str:
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8")
    json.dump(payload, tmp, indent=2)
    tmp.flush()
    tmp.close()
    return tmp.name


def generate_answer(question: str, drug_filter: str, route: str, top_k: int, 
                    temperature: float, show_debug: bool):
    """Generate answer with enhanced UI feedback."""
    
    if not question.strip():
        warning = "⚠️ Please enter a medical question to get started.\n\n" + DISCLAIMER
        return (
            warning,
            "",
            gr.update(visible=False),
            "",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    route_filter = None if route == "any" else route
    drug_to_search = None if drug_filter == "any" else drug_filter

    # Call RAG pipeline
    result = pipeline.answer(
        question,
        top_k=int(top_k),
        temperature=temperature,
        route=route_filter,
        drug=drug_to_search,
    )

    # Format answer with header
    answer_header = "### 💊 Answer\n\n"
    answer_md = answer_header + result["answer"]
    
    # Format sources
    sources_md = format_sources_enhanced(result["sources"])
    
    # Format debug
    debug_md = format_debug_enhanced(result["retrieval_debug"]) if show_debug else ""
    debug_component = gr.update(value=debug_md, visible=show_debug)

    # Build notice with better formatting
    notices = []
    if startup_banner:
        notices.append(startup_banner)
    if result.get("retrieval_notice"):
        notices.append(f"ℹ️ {result['retrieval_notice']}")
    if result.get("backend"):
        backend_label = result["backend"]
        if result.get("used_fallback"):
            backend_label += " (fallback)"
        notices.append(f"🤖 Model: {backend_label}")
    
    notice_md = "\n\n".join(notices) if notices else ""

    # Create download file
    payload_path = _create_download_file(result)
    download = gr.update(value=payload_path, visible=True)
    
    # Show results section
    results_visible = gr.update(visible=True)

    return answer_md, sources_md, debug_component, notice_md, download, results_visible


def load_example(example: str, question_box):
    """Load example query into the question box."""
    return example


# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 1200px !important;
}

.medical-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
}

.medical-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
}

.medical-header p {
    margin: 0.5rem 0 0 0;
    opacity: 0.95;
    font-size: 1.1rem;
}

.example-btn {
    margin: 0.25rem !important;
}

.results-section {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
    margin-top: 1rem;
}

.footer-note {
    text-align: center;
    color: #666;
    font-size: 0.9rem;
    margin-top: 2rem;
    padding: 1rem;
    border-top: 1px solid #e0e0e0;
}
"""


# Build the Gradio interface
with gr.Blocks(css=custom_css, title="Medical RAG Assistant", theme=gr.themes.Soft()) as demo:
    # Header
    gr.HTML("""
        <div class="medical-header">
            <h1>💊 Medical Drug Information Assistant</h1>
            <p>Evidence-based answers from drug monographs with citations</p>
        </div>
    """)
    
    if startup_banner:
        gr.Markdown(f"**{startup_banner}**")
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="🔍 Ask a medical question",
                placeholder="e.g., What are the side effects of azithromycin?",
                lines=3,
                max_lines=5,
            )
            
            # Example queries
            gr.Markdown("**Quick examples:** (click to load)")
            with gr.Row():
                example_buttons = []
                for i, example in enumerate(EXAMPLE_QUERIES[:3]):
                    btn = gr.Button(
                        example[:50] + "..." if len(example) > 50 else example,
                        size="sm",
                        elem_classes="example-btn"
                    )
                    example_buttons.append(btn)
            
            with gr.Row():
                for i, example in enumerate(EXAMPLE_QUERIES[3:]):
                    btn = gr.Button(
                        example[:50] + "..." if len(example) > 50 else example,
                        size="sm",
                        elem_classes="example-btn"
                    )
                    example_buttons.append(btn)
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Search Settings")
            
            drug_filter = gr.Dropdown(
                choices=["any", "azithromycin", "ibuprofen", "ciprofloxacin"],
                value="any",
                label="Drug filter",
                info="Filter results by specific drug"
            )
            
            route = gr.Dropdown(
                choices=["any", "oral", "intravenous", "ophthalmic"],
                value="any",
                label="Route filter",
                info="Filter by administration route"
            )
            
            with gr.Row():
                topk = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Results to retrieve",
                    info="Number of source passages"
                )
                
                temp = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.05,
                    label="Temperature",
                    info="Lower = more focused"
                )
            
            show_debug = gr.Checkbox(
                value=False,
                label="Show retrieval debug info",
                info="View scoring details"
            )
    
    # Search button
    with gr.Row():
        run_button = gr.Button(
            "🔎 Search & Answer",
            variant="primary",
            size="lg",
            scale=2
        )
        clear_button = gr.Button(
            "🗑️ Clear",
            size="lg",
            scale=1
        )
    
    # Results section (initially hidden)
    results_section = gr.Column(visible=False)
    
    with results_section:
        gr.Markdown("---")
        
        # Tabs for organized output
        with gr.Tabs():
            with gr.Tab("💊 Answer"):
                answer_md = gr.Markdown()
                
                with gr.Row():
                    download_button = gr.DownloadButton(
                        label="📥 Download Full Response (JSON)",
                        visible=False,
                        size="sm"
                    )
            
            with gr.Tab("📚 Sources"):
                sources_md = gr.Markdown()
            
            with gr.Tab("ℹ️ System Info"):
                notice_md = gr.Markdown()
            
            with gr.Tab("🔍 Debug", visible=True) as debug_tab:
                debug_md = gr.Markdown(visible=False)
    
    # Footer
    gr.HTML("""
        <div class="footer-note">
            <p><strong>⚠️ Important:</strong> This tool provides information from drug monographs for educational purposes only.</p>
            <p>Always consult a licensed healthcare professional for medical advice and treatment decisions.</p>
        </div>
    """)
    
    # Wire up the example buttons
    for btn, example in zip(example_buttons, EXAMPLE_QUERIES):
        btn.click(
            fn=lambda ex=example: ex,
            inputs=None,
            outputs=question
        )
    
    # Wire up the search button
    run_button.click(
        generate_answer,
        inputs=[question, drug_filter, route, topk, temp, show_debug],
        outputs=[answer_md, sources_md, debug_md, notice_md, download_button, results_section],
    )
    
    # Wire up the clear button
    def clear_all():
        return (
            "",  # question
            gr.update(visible=False),  # results_section
            "",  # answer
            "",  # sources
            gr.update(visible=False),  # debug
            "",  # notice
            gr.update(visible=False),  # download
        )
    
    clear_button.click(
        clear_all,
        outputs=[question, results_section, answer_md, sources_md, debug_md, notice_md, download_button]
    )


if __name__ == "__main__":
    demo.queue(max_size=8)
    from fastapi import FastAPI
    import uvicorn

    fastapi_app = FastAPI()
    fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/")
    print("\n" + "="*60)
    print("🚀 Medical RAG Assistant")
    print("="*60)
    print("📍 URL: http://127.0.0.1:7861")
    print("💡 Tip: Try the example queries to get started!")
    print("="*60 + "\n")
    uvicorn.run(fastapi_app, host="127.0.0.1", port=7861)