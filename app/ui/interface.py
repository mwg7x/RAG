import os
from pathlib import Path
from typing import Any, List

import streamlit as st
import httpx


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def _format_citations(citations: List[dict]) -> str:
    if not citations:
        return ""
    formatted = ""
    for item in citations:
        source = item.get("source", "unknown")
        section = item.get("section", "N/A")
        formatted += f"📄 **{source}** ({section})\n"
    return formatted


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except Exception:
        return response.text


def upload_files(files: List) -> None:
    if not files:
        st.warning("Please upload at least one .pdf or .docx file.")
        return

    prepared = []
    for f in files:
        file_path = Path(f.name)
        prepared.append(("files", (file_path.name, open(file_path, "rb"), "application/octet-stream")))

    try:
        with st.spinner("Ingesting documents..."):
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{API_BASE_URL}/upload", files=prepared)
                response.raise_for_status()
                data = response.json()
                st.success(f"✅ {data.get('message')}")
                st.info(f"📎 Files: {', '.join(data.get('files', []))}")
    except httpx.HTTPStatusError as exc:
        detail = _extract_error_detail(exc.response)
        st.error(f"❌ Upload failed ({exc.response.status_code}): {detail}")
    except Exception as exc:
        st.error(f"❌ Upload failed: {exc}")
    finally:
        for _, (_, file_obj, _) in prepared:
            file_obj.close()


def ask_question(message: str) -> None:
    try:
        with st.spinner("Searching your documents..."):
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{API_BASE_URL}/ask", json={"question": message})
                response.raise_for_status()
                data = response.json()
    except httpx.HTTPStatusError as exc:
        detail = _extract_error_detail(exc.response)
        st.error(f"❌ Request failed ({exc.response.status_code}): {detail}")
        return
    except Exception as exc:
        st.error(f"❌ Request failed: {exc}")
        return

    answer = data.get("answer", "")
    citations = _format_citations(data.get("citations", []))
    
    st.markdown(answer)
    if citations:
        st.markdown("### 📚 Sources")
        st.markdown(citations)


def summarize_contract() -> None:
    try:
        with st.spinner("Summarizing contract..."):
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{API_BASE_URL}/summary")
                response.raise_for_status()
                data = response.json()
    except httpx.HTTPStatusError as exc:
        detail = _extract_error_detail(exc.response)
        st.error(f"❌ Summary failed ({exc.response.status_code}): {detail}")
        return
    except Exception as exc:
        st.error(f"❌ Summary failed: {exc}")
        return

    answer = data.get("answer", "")
    citations = _format_citations(data.get("citations", []))
    
    st.markdown(answer)
    if citations:
        st.markdown("### 📚 Sources")
        st.markdown(citations)


def main():
    # Set page config
    st.set_page_config(
        page_title="Smart Contract Assistant",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS for modern styling
    st.markdown("""
        <style>
            :root {
                --primary-color: #2E86AB;
                --secondary-color: #A23B72;
                --success-color: #06A77D;
            }
            
            .main {
                padding: 2rem;
            }
            
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size: 1.1rem;
                font-weight: 600;
            }
            
            h1 {
                color: #1f1f1f;
                font-size: 2.5rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
            }
            
            .subtitle {
                color: #666;
                font-size: 1.1rem;
                margin-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("📋")
    with col2:
        st.markdown("# Smart Contract Assistant")
        st.markdown("*Analyze, summarize, and ask questions about your contracts powered by AI*", 
                    unsafe_allow_html=True)
    
    st.divider()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📤 Upload Documents", "💬 Ask Questions"])
    
    with tab1:
        st.markdown("### 📄 Upload Your Contracts")
        st.markdown("Supported formats: PDF, DOCX")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("📥 Ingest Documents", use_container_width=True, type="primary"):
                if uploaded_files:
                    upload_files(uploaded_files)
                else:
                    st.warning("Please select at least one file.")
        
        if uploaded_files:
            with st.expander("📋 Selected Files"):
                for file in uploaded_files:
                    st.write(f"• {file.name}")
    
    with tab2:
        st.markdown("### 🤖 Contract Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            user_question = st.text_input(
                "Ask a question",
                placeholder="What are the key terms of this contract?",
                label_visibility="collapsed"
            )
        with col2:
            if st.button("🔍 Ask", use_container_width=True, type="primary"):
                if user_question:
                    ask_question(user_question)
                else:
                    st.warning("Please enter a question.")
        
        st.divider()
        
        if st.button("📊 Generate Summary", use_container_width=True, type="secondary"):
            summarize_contract()


if __name__ == "__main__":
    main()