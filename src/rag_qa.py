"""
rag_qa.py
Connects retrieval (vectorstore.py) to Ollama generation, with grounding,
citations, and an out-of-scope safety guard.
"""

import ollama
from vectorstore import ClinicalVectorStore

OLLAMA_MODEL = "llama2"

ADVICE_TRIGGERS = [
    "should i take", "what should i do", "do i have", "am i at risk",
    "what medication should i", "can you diagnose me", "is it safe for me to"
]

SYSTEM_PROMPT = """You are a clinical document assistant. You answer questions about a SPECIFIC patient's clinical note using ONLY the context provided below.

Rules:
1. Answer using ONLY information present in the provided context. Do not use outside medical knowledge.
2. If the answer is not in the context, say exactly: "I cannot find this information in the document."
3. Be concise and factual. Quote specific values, medication names, and doses exactly as written.
4. You are summarizing a document, not giving medical advice. If asked for advice/diagnosis for the user themselves, decline and explain this tool is for document Q&A only.
5. Never invent information not explicitly stated in the context.
"""


def is_out_of_scope(question: str) -> bool:
    """Returns True if the question is asking for advice about the USER, not the document."""
    question_lower = question.lower()
    for trigger in ADVICE_TRIGGERS:
        if trigger in question_lower:
            return True
    return False


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """Combines retrieved chunks + the question into one prompt for the LLM."""
    context_block = "\n\n---\n\n".join(context_chunks)
    return f"""CONTEXT:
{context_block}

QUESTION:
{question}

Answer using only the context above."""


def answer_question(
    store: ClinicalVectorStore,
    question: str,
    top_k: int = 3,
    note_id: str = None
) -> dict:
    """Full RAG pipeline: check safety -> retrieve -> build prompt -> call LLM -> return answer + sources."""

    if is_out_of_scope(question):
        return {
            "answer": "I can only answer questions about the content of this document. I'm not able to give medical advice. Please consult a healthcare professional.",
            "sources": [],
            "blocked": True,
        }

    results = store.query(question, top_k=top_k, note_id=note_id)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return {
            "answer": "I cannot find this information in the document.",
            "sources": [],
            "blocked": False,
        }

    prompt = build_prompt(question, documents)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    answer_text = response["message"]["content"]

    sources = [
        {"note_id": meta["note_id"], "section": meta["section"]}
        for meta in metadatas
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "blocked": False,
    }


if __name__ == "__main__":
    store = ClinicalVectorStore()
    result = answer_question(store, "What medication was given for chest pain?")
    print("ANSWER:", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  [{s['note_id']} / {s['section']}]")