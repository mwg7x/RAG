import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from pydantic import BaseModel

from app.core.ingestion import DocumentProcessor
from app.core.rag_chain import RAGService


app = FastAPI(title="Smart Contract Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

processor: DocumentProcessor | None = None
rag_service: RAGService | None = None


class QuestionPayload(BaseModel):
    question: str


def _ensure_services() -> tuple[DocumentProcessor, RAGService]:
    global processor, rag_service

    if processor is not None and rag_service is not None:
        return processor, rag_service

    processor = DocumentProcessor(persist_directory=os.getenv("CHROMA_PATH", "./chroma_db"))
    rag_service = RAGService(document_processor=processor)
    return processor, rag_service


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "services_ready": processor is not None and rag_service is not None}


@app.post("/upload")
async def upload_contracts(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        processor, rag_service = _ensure_services()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service initialization failed: {exc}") from exc

    allowed = {".pdf", ".docx"}
    saved_paths: List[str] = []

    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in allowed:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

        destination = UPLOAD_DIR / (file.filename or f"contract{suffix}")
        content = await file.read()
        destination.write_bytes(content)
        saved_paths.append(str(destination))

    try:
        processor.clear_store()
        processor.add_documents(saved_paths)
        rag_service.refresh_retriever()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return {
        "message": "Documents uploaded and indexed successfully",
        "files": [Path(path).name for path in saved_paths],
    }


@app.post("/ask")
def ask_question(payload: QuestionPayload):
    try:
        _, rag_service = _ensure_services()
        result = rag_service.ask({"question": payload.question})
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {exc}") from exc


@app.post("/summary")
def summarize_contract():
    try:
        _, rag_service = _ensure_services()
        return rag_service.summarize({})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summary failed: {exc}") from exc


_DISABLED_LANGSERVE_ENDPOINTS = ["stream", "stream_log", "stream_events"]

if os.getenv("ENABLE_LANGSERVE_ROUTES", "0") == "1":
    try:
        _, rag_service = _ensure_services()
        add_routes(
            app,
            rag_service.rag_runnable,
            path="/rag",
            disabled_endpoints=_DISABLED_LANGSERVE_ENDPOINTS,
        )
        add_routes(
            app,
            rag_service.summary_runnable,
            path="/summary_chain",
            disabled_endpoints=_DISABLED_LANGSERVE_ENDPOINTS,
        )
    except Exception:
        pass

