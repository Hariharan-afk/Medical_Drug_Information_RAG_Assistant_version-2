---
title: Medical Drug Information RAG
emoji: 💊
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.43.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# 💊 Medical Drug Information RAG Assistant

An intelligent medical information retrieval system that answers questions about medications using Retrieval-Augmented Generation (RAG). The system searches through drug monographs and provides evidence-based answers with proper citations.

**Try the Chatbot by following the link: https://huggingface.co/spaces/llSTRIKERll/RAG_based_QA_system**

## 🌟 Features

- **Hybrid Retrieval**: Combines semantic search (FAISS) with lexical search (BM25) for optimal accuracy
- **Multi-drug Support**: Query information across different medications
- **Route-specific Information**: Filter by administration route (oral, IV, ophthalmic, etc.)
- **Source Citations**: Every answer includes references to source documents
- **Smart Filtering**: Drug and route filters with graduated scoring
- **Interactive UI**: Modern interface with example queries and tabbed results

## 🔧 Technology Stack

- **Retrieval**: FAISS (semantic) + BM25 (lexical)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Groq (llama-3.1-8b-instant) with FLAN-T5 fallback
- **Framework**: Gradio 4.43.0
- **Backend**: FastAPI + Uvicorn

## 📊 How It Works

1. **User Query**: Enter a medical question (e.g., "What are the side effects of azithromycin?")
2. **Hybrid Search**: System retrieves relevant passages using semantic + keyword matching
3. **Context Building**: Top passages are formatted with metadata (drug, route, section)
4. **Answer Generation**: LLM generates a concise, cited answer from the retrieved context
5. **Source Display**: Original sources are provided with links for verification

## 🎯 Example Queries

- "What are the side effects of azithromycin?"
- "What is the correct dosage for oral azithromycin in adults?"
- "Can azithromycin be used during pregnancy?"
- "What drugs interact with azithromycin?"
- "What are the differences between oral and IV azithromycin?"

## ⚙️ Configuration

The system uses environment variables for configuration (`.env` file):

```bash
# LLM Configuration
LLM_PROVIDER=openai_compat
LLM_BASE_URL=https://api.groq.com/openai
LLM_API_KEY=your_groq_api_key
MISTRAL_MODEL=llama-3.1-8b-instant

# Retrieval Settings
RETRIEVAL_BM25_ALPHA=0.4  # 40% semantic + 60% BM25
RETRIEVAL_TOPK=5
TIMEOUT_SECONDS=60
```

## 📁 Project Structure

```
.
├── app.py                  # Gradio UI
├── rag.py                  # RAG orchestration
├── retriever.py            # Hybrid retrieval (FAISS + BM25)
├── generator.py            # LLM adapter layer
├── index_builder.py        # FAISS index builder
├── data_ingest.py          # Data loading and normalization
├── prompt_templates.py     # System and user prompts
├── config.py              # Configuration management
├── utils.py               # Helper utilities
├── data/
│   └── chunks.jsonl       # Drug monograph chunks
└── artefacts/
    ├── index.faiss        # FAISS vector index
    └── records.pkl        # Metadata records
```

## 🚀 Local Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd medical-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API key

# Build the index (first time only)
python index_builder.py --rebuild

# Run the application
python app.py
```

Visit `http://127.0.0.1:7861` to use the application.

## 📦 Dependencies

```
gradio==4.43.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
rank-bm25>=0.2.2
httpx>=0.24.0
numpy>=1.24.0
python-dotenv>=1.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
transformers>=4.35.0  # For FLAN-T5 fallback
```

## ⚠️ Important Disclaimers

- **Not Medical Advice**: This tool provides information from drug monographs for educational purposes only
- **Consult Professionals**: Always consult licensed healthcare professionals for medical advice
- **Accuracy**: While the system aims for accuracy, always verify critical information with official sources
- **Limitations**: The system only knows about drugs in its training data (see `data/chunks.jsonl`)

## 🎓 Use Cases

- **Medical Students**: Quick reference for drug information
- **Healthcare Educators**: Teaching tool for pharmacology
- **Researchers**: Exploring drug interactions and contraindications
- **General Public**: Learning about prescribed medications

## 🔬 Model Performance

- **Retrieval Accuracy**: Hybrid approach (40% semantic + 60% BM25) optimized for medical terminology
- **Response Time**: ~2-5 seconds per query (depends on LLM provider)
- **Citation Quality**: All answers include source references
- **Fallback Mode**: Uses local FLAN-T5 if primary LLM unavailable

## 🛠️ Advanced Features

### Debug Mode
Enable "Show retrieval debug info" to see:
- Retrieval scores for each passage
- Drug and section metadata
- Text previews of matched chunks

### Filters
- **Drug Filter**: Focus on specific medications
- **Route Filter**: Get route-specific information (oral, IV, etc.)
- **Top-K**: Adjust number of retrieved passages (3-10)
- **Temperature**: Control response randomness (0.0-1.0)

## 📝 Data Sources

Drug information is sourced from:
- Mayo Clinic drug monographs
- FDA-approved prescribing information
- Peer-reviewed pharmaceutical databases

All sources are cited in responses with clickable links.

## 🤝 Contributing

This is an educational project. For improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## 📄 License

Apache 2.0 License - See LICENSE file for details

## 🙏 Acknowledgments

- Drug monograph data providers (Mayo Clinic, FDA)
- Hugging Face for Sentence Transformers
- Anthropic for guidance on RAG systems
- Groq for fast LLM inference

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Built with ❤️ for better medical information access**




