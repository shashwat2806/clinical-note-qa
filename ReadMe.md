# Clinical Note Q&A — RAG-based Clinical Document Assistant

A fully local Retrieval-Augmented Generation (RAG) pipeline that answers 
natural-language questions about clinical discharge summaries, grounded in 
the actual document text with source citations.

## Why this project

Most "healthcare AI chatbot" demos are thin LLM wrappers with no grounding —
exactly the failure mode that makes hallucination dangerous in clinical settings.
This project instead:

- Retrieves relevant sections from real clinical notes before answering
- Cites which part of the document supported each answer
- Refuses to give medical advice — answers questions about documents, not users
- Runs fully locally (no external API calls, no patient data leaving the machine)

## Architecture

Clinical Notes (.txt)
↓
Section-aware Chunker (chunker.py)

Splits by clinical headers: HPI, Medications, Assessment/Plan, etc.
Preserves patient context in every chunk
↓
Embedding + Vector Store (vectorstore.py)
sentence-transformers (all-MiniLM-L6-v2)
ChromaDB (local, persistent)
↓
RAG QA Pipeline (rag_qa.py)
Retrieves top-k relevant chunks
Grounds generation via strict system prompt
Out-of-scope safety guard
Returns answer + cited sources
↓
Streamlit Chat UI (app.py)

## Tech Stack

- **Chunking**: Custom section-aware regex-based chunker
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`), fully local
- **Vector Store**: ChromaDB (local, persistent)
- **LLM**: Ollama (`llama2`), fully local
- **UI**: Streamlit
- **Data**: 10 synthetic discharge summaries covering pneumonia, MI, 
  stroke, DKA, COPD, appendicitis, UTI, cellulitis, asthma, and heart failure

## Evaluation

Evaluated on a hand-crafted 32-question benchmark spanning:
- Diagnosis questions
- Medication questions  
- Allergy questions
- Lab value questions
- Negation questions ("does NOT have X")
- Reasoning questions (connecting two facts, e.g. "why was drug X used instead of Y")

**Overall accuracy: [FILL IN AFTER EVAL RUN]%**

| Question Type | Accuracy |
|---|---|
| diagnosis | [TBD] |
| medication | [TBD] |
| allergy | [TBD] |
| lab_value | [TBD] |
| negation | [TBD] |
| reasoning | [TBD] |

## Setup

```bash
# Clone the repo
git clone https://github.com/shashwat2806/clinical-note-qa.git
cd clinical-note-qa

# Install dependencies
pip install -r requirements.txt

# Pull Ollama model
ollama pull llama2

# Build the vector index
cd src
python vectorstore.py

# Run the chat UI
streamlit run app.py
```

## Key Design Decisions

**Section-aware chunking over fixed-size**: Clinical notes have predictable 
structure (HPI, Medications, Assessment/Plan). Splitting on semantic boundaries 
keeps related information together, improving retrieval quality.

**Strict grounding prompt**: The system prompt explicitly forbids the LLM from 
using outside knowledge. This causes honest "I cannot find this" responses 
rather than hallucinated answers — the safer failure mode in a clinical context.

**Fully local stack**: No external API calls means no patient data ever leaves 
the machine — a real constraint in healthcare deployment that this architecture 
respects from the start.

