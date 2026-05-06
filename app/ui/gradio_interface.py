import os
import shutil
from pathlib import Path
from typing import Any, List

import gradio as gr

from app.core.ingestion import DocumentProcessor
from app.core.rag_chain import RAGService


UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

processor: DocumentProcessor | None = None
rag_service: RAGService | None = None


def _ensure_services() -> tuple[DocumentProcessor, RAGService]:
    global processor, rag_service

    if processor is not None and rag_service is not None:
        return processor, rag_service

    processor = DocumentProcessor(persist_directory=os.getenv("CHROMA_PATH", "./chroma_db"))
    rag_service = RAGService(document_processor=processor)
    return processor, rag_service


def _format_citations(citations: List[dict]) -> str:
    if not citations:
        return ""

    lines = ["", "Sources"]
    for item in citations:
        source = item.get("source", "unknown")
        section = item.get("section", "N/A")
        lines.append(f"- {source} ({section})")
    return "\n".join(lines)


def upload_files(files: List[Any] | None) -> str:
    if not files:
        return "Please upload at least one PDF or DOCX file."

    saved_paths: List[str] = []
    allowed = {".pdf", ".docx"}

    try:
        current_processor, current_rag = _ensure_services()

        for file in files:
            raw_path = Path(getattr(file, "name", file))
            suffix = raw_path.suffix.lower()
            if suffix not in allowed:
                return f"Unsupported file type: {suffix or 'unknown'}. Use PDF or DOCX."

            destination = UPLOAD_DIR / raw_path.name
            shutil.copyfile(raw_path, destination)
            saved_paths.append(str(destination))

        current_processor.clear_store()
        current_processor.add_documents(saved_paths)
        current_rag.refresh_retriever()

        names = ", ".join(Path(path).name for path in saved_paths)
        return f"Documents uploaded and indexed successfully: {names}"
    except Exception as exc:
        return f"Upload failed: {exc}"


def ask_question(message: str, history: List[dict] | None) -> tuple[str, List[dict]]:
    history = history or []
    if not message or not message.strip():
        return "", history

    history.append({"role": "user", "content": message.strip()})

    try:
        _, current_rag = _ensure_services()
        result = current_rag.ask({"question": message.strip()})
        answer = result.get("answer", "")
        citations = _format_citations(result.get("citations", []))
        response = f"{answer}{citations}"
    except Exception as exc:
        response = f"Request failed: {exc}"

    history.append({"role": "assistant", "content": response})
    return "", history


def summarize_contract(history: List[dict] | None) -> List[dict]:
    history = history or []
    history.append({"role": "user", "content": "Generate a contract summary."})

    try:
        _, current_rag = _ensure_services()
        result = current_rag.summarize({})
        answer = result.get("answer", "")
        citations = _format_citations(result.get("citations", []))
        response = f"{answer}{citations}"
    except Exception as exc:
        response = f"Summary failed: {exc}"

    history.append({"role": "assistant", "content": response})
    return history


CSS = """
:root {
    --accent: #F1B76D;
    --ink: #373B56;
    --bg: #171927;
    --panel: #202337;
    --panel-soft: #292D44;
    --text: #F7F1E8;
    --muted: #B9BBCA;
}

.gradio-container {
    min-height: 100vh;
    background:
        radial-gradient(circle at 15% 0%, rgba(241, 183, 109, 0.14), transparent 32%),
        linear-gradient(135deg, #171927 0%, var(--ink) 100%) !important;
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.main-shell {
    max-width: 1180px;
    margin: 0 auto;
}

.app-title {
    color: var(--text);
    font-size: 2.35rem;
    font-weight: 800;
    line-height: 1.05;
    margin: 0;
}

.app-subtitle {
    color: var(--muted);
    font-size: 1rem;
    margin: 0.65rem 0 0;
}

.brand-mark {
    align-items: center;
    background: linear-gradient(135deg, var(--accent), #ffd89f);
    border-radius: 8px;
    color: var(--ink);
    display: inline-flex;
    font-size: 1.5rem;
    font-weight: 900;
    height: 52px;
    justify-content: center;
    width: 52px;
}

.panel {
    background: rgba(32, 35, 55, 0.88);
    border: 1px solid rgba(241, 183, 109, 0.2);
    border-radius: 8px;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
    padding: 18px;
}

.panel-title {
    color: var(--accent);
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0;
    margin-bottom: 0.7rem;
    text-transform: uppercase;
}

.gradio-container button.primary,
.gradio-container .primary button {
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: var(--ink) !important;
    font-weight: 800 !important;
}

.gradio-container button.secondary {
    background: var(--panel-soft) !important;
    border-color: rgba(241, 183, 109, 0.35) !important;
    color: var(--text) !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container .wrap,
.gradio-container .file-preview,
.gradio-container .block {
    background: var(--panel) !important;
    border-color: rgba(241, 183, 109, 0.22) !important;
    color: var(--text) !important;
}

.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container p {
    color: var(--text) !important;
}

.gradio-container .message {
    border-radius: 8px !important;
}

.gradio-container .message.user {
    background: rgba(241, 183, 109, 0.18) !important;
    color: var(--text) !important;
}

.gradio-container .message.bot {
    background: rgba(55, 59, 86, 0.82) !important;
    color: var(--text) !important;
}

.status-box textarea {
    min-height: 86px !important;
}
"""


def _theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue="orange",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="sm",
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Smart Contract Assistant") as demo:
        with gr.Column(elem_classes=["main-shell"]):
            gr.HTML(
                """
                <div style="display:flex; gap:16px; align-items:center; margin:18px 0 22px;">
                    <div class="brand-mark">S</div>
                    <div>
                        <h1 class="app-title">Smart Contract Assistant</h1>
                        <p class="app-subtitle">Upload contracts, ask focused questions, and generate concise summaries.</p>
                    </div>
                </div>
                """
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes=["panel"]):
                    gr.Markdown("<div class='panel-title'>Documents</div>")
                    files = gr.File(
                        label="Upload PDF or DOCX",
                        file_count="multiple",
                        file_types=[".pdf", ".docx"],
                    )
                    upload_button = gr.Button("Ingest Documents", variant="primary")
                    status = gr.Textbox(
                        label="Status",
                        value="Ready for your contract files.",
                        interactive=False,
                        lines=4,
                        elem_classes=["status-box"],
                    )
                    summary_button = gr.Button("Generate Summary", variant="secondary")

                with gr.Column(scale=2, elem_classes=["panel"]):
                    gr.Markdown("<div class='panel-title'>Analysis</div>")
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=480,
                        buttons=["copy", "copy_all"],
                        avatar_images=(None, None),
                    )
                    with gr.Row():
                        question = gr.Textbox(
                            label="Question",
                            placeholder="What are the payment terms?",
                            scale=5,
                            lines=1,
                            container=False,
                        )
                        ask_button = gr.Button("Ask", variant="primary", scale=1)

            upload_button.click(upload_files, inputs=files, outputs=status)
            ask_button.click(ask_question, inputs=[question, chatbot], outputs=[question, chatbot])
            question.submit(ask_question, inputs=[question, chatbot], outputs=[question, chatbot])
            summary_button.click(summarize_contract, inputs=chatbot, outputs=chatbot)

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=_theme(), css=CSS)


if __name__ == "__main__":
    main()
