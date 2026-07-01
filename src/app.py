"""
app.py
Professional Streamlit chat interface for the Clinical Note Q&A RAG system.
"""

import streamlit as st
from vectorstore import ClinicalVectorStore
from rag_qa import answer_question
import time

# Page configuration
st.set_page_config(
    page_title="Clinical Note Q&A",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
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
</style>
""", unsafe_allow_html=True)

# Title and description
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown('<div class="header-title">🏥 Clinical Note Q&A Assistant</div>', unsafe_allow_html=True)
    st.markdown("*Ask questions about patient clinical discharge summaries. Powered by RAG (Retrieval-Augmented Generation)*")

with col2:
    st.info("✅ **Live** - Ready to query", icon="ℹ️")

# Disclaimer
st.markdown("""
<div class="warning-box">
⚠️ <b>Important Disclaimer</b><br>
This tool is designed for document Q&A only. It does NOT provide medical advice, diagnoses, or treatment recommendations. 
For medical concerns, please consult a qualified healthcare professional. Always verify information with original documents.
</div>
""", unsafe_allow_html=True)

# Initialize the vector store once and cache it
@st.cache_resource
def load_store():
    """Load and cache the vector store to avoid reloading on every rerun."""
    try:
        store = ClinicalVectorStore()
        return store
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

store = load_store()

if store is None:
    st.error("Unable to initialize the RAG system. Please check your setup.")
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.query_count = 0

# Sidebar with instructions and examples
with st.sidebar:
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Ask a Question** - Type any question about the clinical note
    2. **Get Grounded Answer** - The system retrieves relevant sections
    3. **View Sources** - See which document sections informed the answer
    
    ### 💡 Example Questions
    - "What is the patient's diagnosis?"
    - "What medications were prescribed?"
    - "What are the patient's allergies?"
    - "What tests were performed?"
    - "What is the discharge instruction?"
    
    ### ⚙️ System Info
    - **Model**: Llama2 (via Ollama)
    - **Embeddings**: all-MiniLM-L6-v2
    - **Vector DB**: ChromaDB
    - **Framework**: Streamlit
    """)
    
    st.divider()
    
    st.markdown("### 🎯 About This System")
    st.markdown("""
    **RAG Pipeline**: 
    - Chunks clinical notes by section
    - Embeds text semantically
    - Retrieves relevant sections for your question
    - Generates grounded answers from context only
    - Shows citations for transparency
    """)
    
    # Stats
    st.divider()
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions Asked", st.session_state.query_count)
    with col2:
        st.metric("Active Session", "✅" if store else "❌")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
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

# Chat input and processing
st.divider()

if question := st.chat_input("Ask a question about the clinical note...", key="chat_input"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.query_count += 1
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    
    # Get answer from RAG pipeline
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching clinical notes... ⏳"):
            try:
                # Add small delay for visual feedback
                result = answer_question(store, question, top_k=3)
                time.sleep(0.3)
            except Exception as e:
                st.error(f"Error processing question: {e}")
                result = {
                    "answer": "An error occurred while processing your question. Please try again.",
                    "sources": [],
                    "blocked": False
                }
        
        # Display the answer
        st.markdown(result["answer"])
        
        # Show blocking reason if applicable
        if result["blocked"]:
            st.info(
                "⚠️ This question was outside the scope of document Q&A. Please ask about the clinical note content instead.",
                icon="ℹ️"
            )
        
        # Show sources with better formatting
        if result["sources"]:
            with st.expander("📄 View Sources", expanded=True):
                for idx, src in enumerate(result["sources"], 1):
                    st.markdown(
                        f'<div class="source-item"><b>Source {idx}:</b> {src["note_id"]} / {src["section"]}</div>',
                        unsafe_allow_html=True
                    )
    
    # Store assistant response with metadata
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "blocked": result.get("blocked", False)
    })

# Footer
st.divider()
st.markdown("""
---
**Built with**: Streamlit • ChromaDB • Sentence-Transformers • Ollama  
**GitHub**: [shashwat2806/clinical-note-qa](https://github.com/shashwat2806/clinical-note-qa)
""")