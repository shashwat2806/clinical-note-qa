# 🏥 Clinical Note Q&A

**A fully local Retrieval-Augmented Generation (RAG) pipeline that answers natural-language questions about clinical discharge summaries - grounded in the actual document text, with source citations for every answer.**


>This tool is for document Q&A only. It does **not** provide medical advice, diagnoses, or treatment recommendations. Always verify information against original clinical documents and consult a qualified healthcare professional for medical concerns.

---

## Why This Project

Most "healthcare AI chatbot" demos are thin wrappers around an LLM API call, with no grounding mechanism - exactly the failure mode that makes hallucination dangerous in a clinical setting.

This project is built to avoid that failure mode **by design**, not by prompt-level luck:

- 🔍 **Retrieves** relevant sections from real clinical notes *before* generating any answer
- 📌 **Cites** exactly which part of the document supported each answer
- 🚫 **Refuses** to give medical advice — it answers questions about the document's contents, never about patient care
- 🔒 **Runs fully locally** — no external API calls, no patient data ever leaves the machine

---

## How It Works

```
Clinical Notes (.txt)
        │
        ▼
Section-Aware Chunker            (chunker.py)
  • Splits notes by clinical headers: HPI, Medications, Assessment/Plan, etc.
  • Preserves patient context within every chunk
        │
        ▼
Embedding + Vector Store         (vectorstore.py)
  • sentence-transformers (all-MiniLM-L6-v2)
  • ChromaDB — local, persistent
        │
        ▼
RAG QA Pipeline                  (rag_qa.py)
  • Retrieves top-k relevant chunks
  • Grounds generation via a strict system prompt
  • Out-of-scope safety guard — refuses to guess
  • Returns answer + cited sources
        │
        ▼
Streamlit Chat UI                (app.py)
```

Each stage is isolated in its own module, so the chunker, retriever, and generator can each be tested, debugged, and swapped independently.

---

## Tech Stack

| Component      | Technology                                  | Notes                                   |
|-----------------|----------------------------------------------|------------------------------------------|
| Chunking        | Custom section-aware regex chunker           | Splits along semantic boundaries, not fixed size |
| Embeddings      | `sentence-transformers` (`all-MiniLM-L6-v2`) | Fully local, no API calls                |
| Vector Store    | ChromaDB                                     | Local, persistent                        |
| LLM             | Ollama (`llama2`)                            | Fully local generation                   |
| UI              | Streamlit                                    | Chat interface with visible citations    |
| Data            | 10 synthetic discharge summaries             | Pneumonia, MI, stroke, DKA, COPD, appendicitis, UTI, cellulitis, asthma, heart failure |

---

## Evaluation

Evaluated on a hand-crafted **32-question benchmark** spanning six categories: diagnosis, medication, allergy, lab value, negation ("does NOT have X"), and reasoning (connecting two facts, e.g. *"why was drug X used instead of Y"*).

**Overall accuracy: 56.2% (18/32)**

| Question Type | Accuracy |
|----------------|----------|
| Diagnosis      | 100%     |
| Negation       | 83%      |
| Reasoning      | 83%      |
| Allergy        | 50%      |
| Medication     | 33%      |
| Lab Value      | 25%      |

Performance is strongest on questions answerable from a single, prominently placed section (diagnosis, negation, reasoning). Medication and lab-value questions — which often require matching precise numeric or list-style values — are the clearest opportunity for improvement (see [Roadmap](#roadmap)).

---

## Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed locally

### Installation

```bash
# Clone the repo
git clone https://github.com/shashwat2806/clinical-note-qa.git
cd clinical-note-qa

# Install dependencies
pip install -r requirements.txt

# Pull the local LLM
ollama pull llama2
```

### Run

```bash
# Build the vector index (first run only, or after changing notes)
cd src
python vectorstore.py

# Launch the chat UI
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Project Structure

```
clinical-note-qa/
├── data/
│   ├── notes/          # Source clinical discharge summaries (.txt)
│   └── eval/           # 32-question evaluation benchmark
├── src/
│   ├── app.py           # Streamlit chat interface
│   ├── chunker.py        # Section-aware document chunking
│   ├── vectorstore.py    # Embedding + ChromaDB indexing
│   ├── rag_qa.py         # Retrieval + grounded generation pipeline
│   └── evaluate.py       # Benchmark evaluation script
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

**Section-aware chunking over fixed-size chunking** — Clinical notes have predictable structure (HPI, Medications, Assessment/Plan). Splitting on semantic boundaries keeps related information together in a single chunk, directly improving retrieval quality over naive fixed-size windowing.

**Strict grounding prompt** — The system prompt explicitly forbids the LLM from using outside knowledge. This produces honest *"I cannot find this information in the document"* responses instead of hallucinated answers — the safer failure mode in a clinical context, even at some cost to recall.

**Fully local stack** — No external API calls means no patient data ever leaves the machine, a real constraint in healthcare deployments that this architecture respects from the start rather than bolting on later.

---

## Roadmap

- [ ] Improve medication / lab-value accuracy with a stronger or fine-tuned local LLM
- [ ] Add a re-ranking step to improve precision on multi-fact reasoning questions
- [ ] Expand the evaluation benchmark beyond 32 questions
- [ ] Support multi-document, multi-patient querying with stronger context isolation

---
