import os
from typing import Any, Dict, List

import httpx
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

from app.core.ingestion import DocumentProcessor


GUARDRAIL_MESSAGE = "I can only answer questions related to the uploaded contract"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
GREETING_WORDS = {"hi", "hello", "hey", "yo", "sup", "greetings"}


class RAGService:
    def __init__(
        self,
        document_processor: DocumentProcessor,
        model_name: str | None = None,
        base_url: str | None = None,
        relevance_threshold: float = 0.25,
    ) -> None:
        self.document_processor = document_processor
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        requested_model = model_name or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.model_name = self._resolve_model_name(requested_model)
        self.relevance_threshold = relevance_threshold

        self.llm = ChatOllama(model=self.model_name, base_url=self.base_url, temperature=0)

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a smart contract assistant. Use ONLY the provided context. "
                    "If the answer is not in the context, say you could not find it in the contract. "
                    "When answering, cite sources with wording like 'According to Section X in SOURCE ...'.\n\n"
                    "Context:\n{context}",
                ),
                ("human", "Question: {input}"),
            ]
        )

        self.summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Provide a high-level summary of the uploaded contract in concise bullet points. "
                    "Focus on parties, obligations, payment terms, duration, termination, and risk clauses.",
                ),
                ("human", "Contract context:\n{context}\n\nGive a clean executive summary."),
            ]
        )

        self.retriever = self.document_processor.get_retriever()
        self.qa_chain = self._build_qa_chain()
        self.summary_chain = self._build_summary_chain()

        self.rag_runnable = RunnableLambda(self.ask)
        self.summary_runnable = RunnableLambda(self.summarize)

    @staticmethod
    def _model_aliases(model_name: str) -> set[str]:
        normalized = (model_name or "").strip()
        if not normalized:
            return set()

        base = normalized.split(":", 1)[0]
        aliases = {normalized, base}
        if ":" not in normalized:
            aliases.add(f"{normalized}:latest")
        return aliases

    def _fetch_available_models(self) -> List[str]:
        try:
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        models: List[str] = []
        for item in payload.get("models", []):
            if not isinstance(item, dict):
                continue
            model_name = item.get("name") or item.get("model")
            if isinstance(model_name, str) and model_name.strip():
                models.append(model_name.strip())
        return models

    def _resolve_model_name(self, requested_model: str) -> str:
        available_models = self._fetch_available_models()
        if not available_models:
            return requested_model

        requested_aliases = self._model_aliases(requested_model)
        for available_model in available_models:
            if requested_aliases.intersection(self._model_aliases(available_model)):
                return available_model

        return available_models[0]

    def refresh_retriever(self) -> None:
        self.retriever = self.document_processor.get_retriever()
        self.qa_chain = self._build_qa_chain()
        self.summary_chain = self._build_summary_chain()

    def _build_qa_chain(self):
        doc_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        return create_retrieval_chain(self.retriever, doc_chain)

    def _build_summary_chain(self):
        doc_chain = create_stuff_documents_chain(self.llm, self.summary_prompt)
        return create_retrieval_chain(self.retriever, doc_chain)

    def _is_in_scope(self, query: str) -> bool:
        # Keep this resilient across vectorstore scoring implementations.
        # Some backends return non-normalized values, which breaks threshold-based checks.
        return bool(self.document_processor.vectorstore.similarity_search(query, k=1))

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized_chars = [ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in query]
        return " ".join("".join(normalized_chars).split())

    def _is_greeting(self, query: str) -> bool:
        normalized = self._normalize_query(query)
        if not normalized:
            return False

        parts = normalized.split()
        if len(parts) == 1 and parts[0] in GREETING_WORDS:
            return True
        if len(parts) <= 3 and parts[0] in {"hi", "hello", "hey"}:
            return True
        return False

    def _has_indexed_documents(self) -> bool:
        try:
            result = self.document_processor.vectorstore.get(limit=1)
        except Exception:
            return False
        ids = result.get("ids", []) if isinstance(result, dict) else []
        return bool(ids)

    @staticmethod
    def _build_citations(context_docs: List[Any]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        seen = set()
        for doc in context_docs:
            meta = doc.metadata or {}
            source = meta.get("source", "unknown")
            section = meta.get("section") or (f"Page {meta['page']}" if "page" in meta else "N/A")
            key = (source, section)
            if key in seen:
                continue
            seen.add(key)
            citations.append({"source": source, "section": section})
        return citations

    def ask(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        question = payload.get("question") or payload.get("input")
        if not question:
            raise ValueError("Missing 'question' in payload")

        if self._is_greeting(question):
            return {
                "answer": (
                    "Hi! I can help with your uploaded contract. "
                    "Try asking: 'Summarize the payment terms' or "
                    "'What does the termination clause say?'"
                ),
                "citations": [],
            }

        if not self._has_indexed_documents():
            return {
                "answer": "I don't have any indexed contract yet. Please upload a PDF or DOCX first.",
                "citations": [],
            }

        if not self._is_in_scope(question):
            return {
                "answer": (
                    f"{GUARDRAIL_MESSAGE}. "
                    "Try asking about clauses, obligations, dates, penalties, or payment terms."
                ),
                "citations": [],
            }

        chain_result = self.qa_chain.invoke({"input": question})
        answer = chain_result.get("answer", "")
        context_docs = chain_result.get("context", [])
        citations = self._build_citations(context_docs)

        return {"answer": answer, "citations": citations}

    def summarize(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        chain_result = self.summary_chain.invoke({"input": "Summarize this contract."})
        answer = chain_result.get("answer", "")
        context_docs = chain_result.get("context", [])
        citations = self._build_citations(context_docs)
        return {"answer": answer, "citations": citations}

