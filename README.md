# Drug Monograph RAG (Ibuprofen) — Mistral API Edition

This repository hosts a production-ready retrieval-augmented generation (RAG) stack for answering ibuprofen monograph questions. Retrieval runs over `data/chunks.jsonl`; generation uses an open-source Mistral model exposed via HTTP (OpenAI-compatible or Ollama). A CPU fallback (`google/flan-t5-base`) keeps the app functional when no remote LLM is available. The project ships with a Gradio UI suitable for local development and deployment to a Hugging Face Space.

## Fill These Before Running

Search for the markers `# === FILL_ME` and update the following:

1. `.env` (copy from `.env.example`):
   - `LLM_PROVIDER`
   - `LLM_BASE_URL`
   - `LLM_API_KEY`
   - `LLM_API_KEY_HEADER`
   - `MISTRAL_MODEL`
2. `config.py`: verify the defaults for `LLM_BASE_URL`, `LLM_API_KEY`, and `MISTRAL_MODEL`.
3. Any additional files with `# === FILL_ME` comments (search the repo to confirm nothing was missed).

Without valid values the app will fall back to the local FLAN-T5 model and warn you at startup.

## Quick Start (Local)

```bash
python -m venv .venv
. .venv/Scripts/activate  # on Windows; use `. .venv/bin/activate` on Unix
pip install -r requirements.txt

# Optional: copy environment template
cp .env.example .env  # edit placeholders afterwards

# Build the FAISS index (first run may take several minutes)
python index_builder.py --rebuild

# Launch the Gradio app
python app.py
```

Open the provided local URL to query the assistant. If the banner reports “Using local FLAN-T5 fallback (CPU)”, update your `.env` and restart once a remote Mistral endpoint is ready.

## Configure the Mistral Backend

The generator supports two primary adapters plus the local fallback:

1. **OpenAI-compatible gateway** (e.g., vLLM, TGI, OpenRouter, HF Inference Endpoint)
   - `.env`:
     ```env
     LLM_PROVIDER=openai_compat
     LLM_BASE_URL=https://YOUR-LLM-ENDPOINT  # === FILL_ME
     LLM_API_KEY=YOUR-API-KEY               # === FILL_ME (leave blank if not needed)
     LLM_API_KEY_HEADER=Authorization       # Adjust if your endpoint expects a different header
     MISTRAL_MODEL=mistral-7b-instruct      # === FILL_ME (model id exposed by your gateway)
     ```
   - The app will call `${LLM_BASE_URL}/v1/chat/completions` with standard OpenAI payloads.

2. **Ollama** (local inference)
   - `.env`:
     ```env
     LLM_PROVIDER=ollama
     LLM_BASE_URL=http://localhost:11434   # === FILL_ME if different
     MISTRAL_MODEL=mistral                 # === FILL_ME
     ```
   - Ensure `ollama serve` is running with the desired Mistral model pulled.

3. **Fallback** — if the selected backend fails or remains unconfigured, the router automatically switches to `google/flan-t5-base` on CPU.

## Retrieval & Indexing

- Source knowledge base: `data/chunks.jsonl` (not committed; copy your generated file into this path).
- The index builder writes artefacts under `artefacts/` (`index.faiss`, `records.pkl`). Keep this directory ignored by git.
- Rebuild command (customized paths supported):
  ```bash
  python index_builder.py --rebuild --chunks-path data/chunks.jsonl \
      --index-path artefacts/index.faiss \
      --records-path artefacts/records.pkl
  ```

## Running Tests

The repository includes lightweight smoke tests that stub heavy components:

```bash
pytest -q
```

## Deploying to a Hugging Face Space

1. Create a new Space (Gradio SDK) and push the repo contents, including `huggingface.yaml`.
2. In the Space Settings → Variables and secrets, add:
   - `LLM_PROVIDER`
   - `LLM_BASE_URL`
   - `LLM_API_KEY` (as a **secret**)
   - `LLM_API_KEY_HEADER`
   - `MISTRAL_MODEL`
3. Upload your `data/chunks.jsonl` and (optionally) pre-built `artefacts/` folder, or allow the Space to build the index on first launch.
4. Deploy. A banner will indicate if the app is using the FLAN fallback.

## Repository Layout

- `app.py` – Gradio interface and deployment entrypoint.
- `config.py` – Central configuration with environment fallbacks.
- `data_ingest.py` – JSONL loader and normalizer.
- `index_builder.py` – FAISS index builder/loader.
- `retriever.py` – Hybrid semantic/BM25 retriever with route filters.
- `generator.py` – Mistral HTTP adapters and local FLAN fallback.
- `rag.py` – Retrieval-generation orchestration and safety guardrails.
- `prompt_templates.py` – System and user prompt templates.
- `tests/` – Smoke tests covering ingestion, indexing, and pipeline wiring.

## Troubleshooting

- **Slow startup**: The first launch downloads the embedding and FLAN models. Subsequent runs use cached weights.
- **Placeholder warnings**: Update `.env` and re-run; the banner disappears once a valid endpoint is configured.
- **Empty answers**: Ensure `data/chunks.jsonl` contains the expected ibuprofen sections and rebuild the index.

Always review generated answers with a qualified clinician before acting on them.

