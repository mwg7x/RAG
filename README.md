# Smart Contract Assistant

A local AI assistant for uploading contract documents, indexing their contents, and asking questions about clauses, payment terms, dates, obligations, termination language, and other contract details.

The app now runs with a dark Gradio interface using:

- `#F1B76D` for the accent color
- `#373B56` for the dark UI base

## Features

- Upload one or more `.pdf` or `.docx` contract files
- Index documents locally with Chroma
- Ask contract-focused questions
- Generate a concise contract summary
- Show source citations from the uploaded documents
- Use a local Ollama model through LangChain

## Requirements

- Python 3.11 or newer
- Ollama installed and running
- An Ollama model downloaded locally

The default model is set in `app/core/rag_chain.py`:

```python
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
```

To download that model:

```bash
ollama pull llama3.2:latest
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama is running:

```bash
ollama serve
```

In another terminal, run the app:

```bash
python main.py
```

Open the Gradio UI:

```text
http://127.0.0.1:7860
```

## Configuration

You can override the Ollama model without editing code:

```bash
set OLLAMA_MODEL=llama3.2:latest
```

You can also override the Ollama server URL:

```bash
set OLLAMA_BASE_URL=http://localhost:11434
```

Optional paths:

```bash
set UPLOAD_DIR=./uploads
set CHROMA_PATH=./chroma_db
```

For development without real embedding downloads, use fake deterministic embeddings:

```bash
set USE_FAKE_EMBEDDINGS=1
```

## Project Structure

```text
main.py                     Gradio app launcher
app/ui/gradio_interface.py  Main Gradio dark UI
app/ui/interface.py         Older Streamlit UI
app/core/ingestion.py       PDF/DOCX loading, chunking, and Chroma indexing
app/core/rag_chain.py       Ollama model setup and RAG question/summary logic
app/api/server.py           Optional FastAPI API
uploads/                    Uploaded documents
chroma_db/                  Local Chroma vector database
```

## API Mode

The FastAPI server is still available if needed:

```bash
uvicorn app.api.server:app --host 127.0.0.1 --port 8000
```

API endpoints:

- `GET /health`
- `POST /upload`
- `POST /ask`
- `POST /summary`

## Notes

The assistant is designed to answer based only on uploaded contract content. If no documents are indexed, upload a PDF or DOCX first.
