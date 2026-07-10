# Clinical Note Q&A — RAG-based Clinical Document Assistant

A fully local Retrieval-Augmented Generation (RAG) pipeline that answers
natural-language questions about clinical discharge summaries, grounded in
the actual document text with source citations. No external API calls —
everything runs on your machine.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Local-green)
![Ollama](https://img.shields.io/badge/Ollama-llama2-orange)
![Tests](https://img.shields.io/badge/Tests-14%20passing-brightgreen)

---

## Why this project

Most "healthcare AI chatbot" demos are thin LLM wrappers with no grounding —
exactly the failure mode that makes hallucination dangerous in clinical settings.
This project instead:

- Retrieves relevant sections from real clinical notes **before** answering
- Cites which part of the document supported each answer
- Refuses to give medical advice — answers questions about documents, not users
- Supports **PDF upload** — upload any discharge summary and chat with it instantly
- Supports **single-note filtering** — scope queries to one patient instead of searching all
- Runs fully locally — no patient data ever leaves the machine

---

## Screenshots

**Main interface**
![Main interface](# Clinical Note Q&A — RAG-based Clinical Document Assistant

A fully local Retrieval-Augmented Generation (RAG) pipeline that answers
natural-language questions about clinical discharge summaries, grounded in
the actual document text with source citations. No external API calls —
everything runs on your machine.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Local-green)
![Ollama](https://img.shields.io/badge/Ollama-llama2-orange)
![Tests](https://img.shields.io/badge/Tests-14%20passing-brightgreen)

---

## Why this project

Most "healthcare AI chatbot" demos are thin LLM wrappers with no grounding —
exactly the failure mode that makes hallucination dangerous in clinical settings.
This project instead:

- Retrieves relevant sections from real clinical notes **before** answering
- Cites which part of the document supported each answer
- Refuses to give medical advice — answers questions about documents, not users
- Supports **PDF upload** — upload any discharge summary and chat with it instantly
- Supports **single-note filtering** — scope queries to one patient instead of searching all
- Runs fully locally — no patient data ever leaves the machine

---

## Screenshots

**Main interface**
![Main interface](docs/screenshot-main.png)

**Note selector — filter queries to a specific patient**
![Note selector](<img width="1895" height="827" alt="docsscreenshot-sidebar png" src="https://github.com/user-attachments/assets/3dc75eeb-4580-4321-997f-78829344e889" />
)

**Grounded Q&A with cited sources**
![Q&A example](<img width="958" height="312" alt="Screenshot 2026-07-09 155727" src="https://github.com/user-attachments/assets/dc68bbb0-a95f-4fa3-8799-d0c70dc5971b" />
)

---

## Architecture)

**Note selector — filter queries to a specific patient**
![Note selector](<img width="1918" height="893" alt="Screenshot 2026-07-09 155742" src="https://github.com/user-attachments/assets/3ad6baae-a146-407f-b12c-b1eac814a001" />
)

---

## Architecture

Clinical Notes (.txt / PDF upload)
↓
Section-aware Chunker (chunker.py)
Splits by clinical headers: HPI, Medications, Assessment/Plan, etc.
Preserves patient context in every chunk
↓
Embedding + Vector Store (vectorstore.py)
sentence-transformers (all-MiniLM-L6-v2), local
ChromaDB (local, persistent)
Optional per-note filtering
↓
RAG QA Pipeline (rag_qa.py)
Retrieves top-k relevant chunks
Grounds generation via strict system prompt
Out-of-scope safety guard
Returns answer + cited sources
↓
Streamlit Chat UI (app.py)
PDF upload in sidebar
Note selector dropdown
Chat interface with source citations
Session tracking

---

## Tech Stack

| Component | Tool |
|---|---|
| Chunking | Custom section-aware regex chunker |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (local, persistent) |
| LLM | Ollama (`llama2`, fully local) |
| PDF Extraction | `pdfplumber` |
| UI | Streamlit |
| Testing | `unittest` / `pytest` (14 tests, chunker module) |

---

## Evaluation

Evaluated on a hand-crafted 32-question benchmark spanning 6 clinical
question types, designed to test both factual retrieval and harder reasoning:

**Overall accuracy: 56.2% (18/32)**

| Question Type | Accuracy | Notes |
|---|---|---|
| diagnosis | 100% | Direct lookup |
| history | 100% | Direct lookup |
| instruction | 100% | Direct lookup |
| negation | 83% | "Patient does NOT have X" — hardest NLP task |
| reasoning | 83% | Multi-fact inference |
| allergy | 50% | |
| medication | 33% | Conservative grounding causes paraphrasing |
| procedure | 33% | Conservative grounding causes paraphrasing |
| lab_value | 25% | Exact number matching penalised by keyword scorer |

**Key finding**: Negation (83%) and reasoning (83%) - the clinically
highest-risk categories - outperform direct factual lookups. The lower
scores for lab values and medications reflect the model's conservative
grounding behavior (paraphrasing rather than quoting exact values), which
is actually the safer failure mode in a clinical context.

---

## Project Structure

clinical-note-qa/
├── data/
│   ├── notes/          # 10 synthetic discharge summaries
│   └── eval/           # 32-question ground truth eval set
├── src/
│   ├── chunker.py      # Section-aware chunking
│   ├── vectorstore.py  # Embeddings + ChromaDB (with per-note filtering)
│   ├── rag_qa.py       # Grounded generation + safety guard
│   ├── pdf_reader.py   # PDF text extraction
│   ├── app.py          # Streamlit chat UI
│   └── evaluate.py     # Evaluation script
├── tests/
│   └── test_chunker.py # 14 unit tests
├── docs/                # Screenshots
├── outputs/             # Evaluation results (generated)
├── Makefile
├── .gitignore
├── requirements.txt
└── README.md

---

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

Or, if you have `make` available:
```bash
make index
make run
```

---

## Usage

**Pre-loaded notes**: Select a specific patient from the sidebar dropdown,
or search across all 10 synthetic discharge summaries.

**PDF upload**: Upload any clinical discharge summary PDF using the sidebar
uploader. The system extracts text, chunks it by section, and makes it
immediately available for Q&A.

**Run tests**:
```bash
make test
# or
python -m pytest tests/ -v
```

**Run evaluation**:
```bash
make eval
# or
cd src && python evaluate.py
```

---

## Key Design Decisions

**Section-aware chunking over fixed-size**: Clinical notes have predictable
structure (HPI, Medications, Assessment/Plan). Splitting on semantic
boundaries keeps related information together, improving retrieval quality
vs naive character-count chunking.

**Strict grounding prompt**: The system prompt explicitly forbids the LLM
from using outside knowledge. This produces honest "I cannot find this"
responses rather than hallucinated answers — the safer failure mode in
a clinical context.

**Per-note filtering**: Retrieval can be scoped to a single patient note
via ChromaDB metadata filtering, avoiding cross-patient contamination when
the user's intent is clearly about one specific document.

**Fully local stack**: No external API calls means no patient data ever
leaves the machine — a real constraint in healthcare deployment that this
architecture respects from the start.

**PDF support**: `pdfplumber` extracts text from uploaded PDFs, feeding
directly into the existing chunker — no changes to the RAG pipeline needed.

---

## Known Limitations

- Vague or ambiguous queries (e.g. a name with no question) can produce
  inconsistent results, since retrieval has no clear signal to anchor on.
  The system performs best with specific, well-formed questions.
- The evaluation script uses keyword-overlap scoring, which under-counts
  correct answers that are paraphrased rather than quoted verbatim
  (particularly affecting lab value and medication questions).
- PDF extraction depends on the PDF containing real text layers; scanned
  image-only PDFs will not extract correctly.

---
