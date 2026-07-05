"""
app.py
Streamlit chat interface for the Clinical Note Q&A RAG system.
Supports pre-loaded synthetic notes, note selector, and PDF upload.
"""

import streamlit as st
import hashlib
import time
from vectorstore import ClinicalVectorStore
from rag_qa import answer_question

st.set_page_config(
    page_title="Clinical Note Q&A",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .source-item {
        background-color: #e8f4f8;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #0088cc;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown('<div class="header-title">🏥 Clinical Note Q&A Assistant</div>', unsafe_allow_html=True)
    st.markdown("*Ask questions about patient clinical discharge summaries. Powered by RAG.*")
with col2:
    st.info("✅ **Live** - Ready to query", icon="ℹ️")

st.markdown("""
<div class="warning-box">
⚠️ <b>Important Disclaimer</b><br>
This tool is designed for document Q&A only. It does NOT provide medical advice,
diagnoses, or treatment recommendations. Always consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_store():
    try:
        return ClinicalVectorStore()
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None


store = load_store()

if store is None:
    st.error("Unable to initialize the RAG system. Please check your setup.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.query_count = 0
    st.session_state.selected_note_id = None

# Sidebar
with st.sidebar:

    # PDF Upload — top of sidebar
    st.markdown("### 📂 Upload a Clinical Note")
    st.markdown("Upload a PDF discharge summary to chat with it directly.")

    uploaded_pdf = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a clinical discharge summary in PDF format"
    )

    if uploaded_pdf is not None:
        from pdf_reader import extract_text_from_pdf, is_valid_pdf_text
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_pdf)

        if not is_valid_pdf_text(pdf_text):
            st.error("Could not extract text. This may be a scanned image PDF.")
        else:
            if st.session_state.get("pdf_filename") != uploaded_pdf.name:
                st.session_state.pdf_text = pdf_text
                st.session_state.pdf_filename = uploaded_pdf.name
                st.session_state.messages = []
                store.reset()
                from chunker import chunk_note
                note_id = hashlib.md5(uploaded_pdf.name.encode()).hexdigest()[:8]
                chunks = chunk_note(note_id, pdf_text)
                store.index_chunks(chunks)
            st.success(f"✅ Loaded: {uploaded_pdf.name}")
            st.caption(f"Extracted {len(pdf_text)} characters")

    st.divider()

    # Note selector
    st.markdown("### 🗂️ Select a Patient Note")
    st.markdown("Filter queries to a specific note, or search across all.")

    NOTE_OPTIONS = {
        "All notes (search across everything)": None,
        "note_01 — J. Mehta (Pneumonia)": "note_01",
        "note_02 — R. Sharma (MI / Chest Pain)": "note_02",
        "note_03 — A. Khan (Asthma, Paediatric)": "note_03",
        "note_04 — S. Verma (UTI / Delirium)": "note_04",
        "note_05 — P. Nair (Appendicitis)": "note_05",
        "note_06 — M. Iyer (Stroke)": "note_06",
        "note_07 — D. Banerjee (COPD)": "note_07",
        "note_08 — T. Reddy (Diabetic Ketoacidosis)": "note_08",
        "note_09 — K. Joshi (Cellulitis)": "note_09",
        "note_10 — L. Fernandes (Heart Failure)": "note_10",
    }

    selected_label = st.selectbox(
        "Choose a note:",
        options=list(NOTE_OPTIONS.keys()),
        index=0,
    )
    st.session_state.selected_note_id = NOTE_OPTIONS[selected_label]

    if st.session_state.selected_note_id:
        st.caption(f"Filtering to: {st.session_state.selected_note_id}")
    else:
        st.caption("Searching across all 10 notes")

    st.divider()

    # How to use
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Upload a PDF above, or select a pre-loaded note
    2. **Ask a Question** in the chat box below
    3. **View Sources** to see which sections were used
    """)

    st.markdown("### 💡 Example Questions")
    st.markdown("""
    - "What is the patient's diagnosis?"
    - "What medications were prescribed?"
    - "What are the patient's allergies?"
    - "What tests were performed?"
    """)

    st.divider()

    # System info
    st.markdown("### ⚙️ System Info")
    st.markdown("""
    - **Model**: Llama2 (via Ollama)
    - **Embeddings**: all-MiniLM-L6-v2
    - **Vector DB**: ChromaDB
    - **Framework**: Streamlit
    """)

    st.divider()

    # Session stats
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions Asked", st.session_state.query_count)
    with col2:
        st.metric("Active Session", "✅" if store else "❌")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("📄 View Sources", expanded=False):
                    for src in message["sources"]:
                        st.markdown(
                            f'<div class="source-item"><b>{src["note_id"]}</b> / {src["section"]}</div>',
                            unsafe_allow_html=True
                        )
            if message.get("blocked", False):
                st.info("⚠️ This question was outside the scope of document Q&A.", icon="ℹ️")

st.divider()

# Chat input
if question := st.chat_input("Ask a question about the clinical note...", key="chat_input"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.query_count += 1

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching clinical notes..."):
            try:
                result = answer_question(
                    store,
                    question,
                    top_k=3,
                    note_id=st.session_state.get("selected_note_id")
                )
                time.sleep(0.3)
            except Exception as e:
                st.error(f"Error: {e}")
                result = {
                    "answer": "An error occurred. Please try again.",
                    "sources": [],
                    "blocked": False
                }

        st.markdown(result["answer"])

        if result["blocked"]:
            st.info("⚠️ This question was outside the scope of document Q&A.", icon="ℹ️")

        if result["sources"]:
            with st.expander("📄 View Sources", expanded=True):
                for idx, src in enumerate(result["sources"], 1):
                    st.markdown(
                        f'<div class="source-item"><b>Source {idx}:</b> {src["note_id"]} / {src["section"]}</div>',
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "blocked": result.get("blocked", False)
    })

# Footer
st.divider()
st.markdown("""
**Built with**: Streamlit • ChromaDB • Sentence-Transformers • Ollama
**GitHub**: [shashwat2806/clinical-note-qa](https://github.com/shashwat2806/clinical-note-qa)
""")