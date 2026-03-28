# Azure RAG AI Search

A **Retrieval-Augmented Generation (RAG)** pipeline built with **Azure Functions (Python)**, **Azure AI Search**, and **Azure OpenAI**. Upload documents to Azure Blob Storage, automatically extract, chunk, embed, and index them — then ask natural language questions and get grounded answers.

## Architecture

```
Documents (PDF/DOCX/PPTX/Images)
        │
        ▼
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Blob Storage    │────▶│  Extractor   │────▶│   Chunker    │
│  (Upload files)  │     │  (OCR/Text)  │     │  (1024 tok)  │
└─────────────────┘     └──────────────┘     └──────┬───────┘
                                                     │
                                                     ▼
                                              ┌──────────────┐
                                              │   Embedder   │
                                              │ (Ada / Small)│
                                              └──────┬───────┘
                                                     │
                                                     ▼
                                            ┌────────────────┐
                                            │  Azure AI      │
                                            │  Search Index  │
                                            └───────┬────────┘
                                                    │
                          User Question ───▶  Vector Search
                                                    │
                                                    ▼
                                            ┌────────────────┐
                                            │  Azure OpenAI  │
                                            │   (GPT-4o)     │
                                            └────────────────┘
                                                    │
                                                    ▼
                                              Grounded Answer
```

## API Endpoints

### `POST /api/index`
Index documents from Azure Blob Storage into the search index.

| Request Body | Description |
|---|---|
| `{}` | Index **all** blobs in the container |
| `{ "blob_path": "folder/file.pdf" }` | Index a **single** blob |

### `POST /api/ask`
Ask a question and get a RAG-powered answer.

```json
{
  "question": "What is supervised learning?",
  "top_k": 4,
  "topic": "ML"
}
```

**Response:**
```json
{
  "answer": "Supervised learning is ...",
  "sources": [
    { "source_link": "...", "score": 0.89, "file_id": "..." }
  ]
}
```

## Tech Stack

| Component | Service |
|---|---|
| Compute | Azure Functions (Python v2) |
| Storage | Azure Blob Storage |
| Search | Azure AI Search (vector search) |
| LLM | Azure OpenAI (GPT-4o) |
| Embeddings | Azure OpenAI (text-embedding-3-small) |
| Document Parsing | Azure Form Recognizer, PyMuPDF, Tesseract OCR |

## Project Structure

```
azure-rag-app/
├── function_app.py                # Azure Functions entry point (HTTP triggers)
├── host.json                      # Azure Functions host configuration
├── requirements.txt               # Python dependencies
├── local.settings.json.example    # Template for environment variables
└── shared/
    ├── blob_client.py             # Azure Blob Storage operations
    ├── extractor.py               # Document text extraction (PDF, DOCX, PPTX, images)
    ├── chunker.py                 # Text chunking with overlap
    ├── embedder.py                # Azure OpenAI embedding generation
    ├── search_client.py           # Azure AI Search client (CRUD + vector search)
    ├── index_manager.py           # Search index creation and management
    ├── indexer_service.py         # Orchestrates extract → chunk → embed → index
    └── rag_service.py             # RAG orchestrator (embed → retrieve → generate)
```

## Getting Started

### Prerequisites
- Python 3.9+
- [Azure Functions Core Tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local)
- Azure subscription with the following services:
  - Azure Blob Storage
  - Azure AI Search
  - Azure OpenAI
  - Azure Form Recognizer (optional, for advanced document parsing)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tharunshaiva/azure-rag-ai-search.git
   cd azure-rag-ai-search
   ```

2. **Create a virtual environment**
   ```bash
   cd azure-rag-app
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp local.settings.json.example local.settings.json
   ```
   Edit `local.settings.json` and fill in your Azure credentials.

5. **Run locally**
   ```bash
   func start
   ```

## Security

> ⚠️ **Never commit `local.settings.json`** — it contains real API keys. The `.gitignore` is configured to exclude it. Use `local.settings.json.example` as a reference template.

## License

This project is for educational/assignment purposes.
