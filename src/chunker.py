"""
chunker.py
Section-aware chunking for clinical discharge summaries.

Why section-aware instead of naive fixed-size chunking?
Clinical notes have a predictable structure (HPI, Medications, Assessment, etc.).
Splitting on these boundaries keeps related information together instead of
cutting a sentence in half at an arbitrary character count.
"""

import os
import re
from dataclasses import dataclass

SECTION_HEADERS = [
    "CHIEF COMPLAINT",
    "HISTORY OF PRESENT ILLNESS",
    "PAST MEDICAL HISTORY",
    "MEDICATIONS ON ADMISSION",
    "PHYSICAL EXAM",
    "LABS AND DIAGNOSTICS",
    "LABS",
    "ASSESSMENT AND PLAN",
    "MEDICATIONS AT DISCHARGE",
    "ALLERGIES",
    "DISCHARGE INSTRUCTIONS",
    "DISCHARGE DIAGNOSIS",
]


@dataclass
class Chunk:
    note_id: str
    section: str
    text: str
    chunk_id: str


def extract_patient_header(text: str) -> str:
    """Pulls the Patient/Age/Sex lines at the top of the note for context."""
    lines = text.strip().splitlines()
    header_lines = []
    prefixes = ("Patient:", "Age:", "Sex:", "Date of Admission:", "Date of Discharge:")

    for line in lines:
        line = line.strip()
        if line.startswith(prefixes):
            header_lines.append(line)
        elif header_lines and not line:
            break

    return " | ".join(header_lines)


def _build_header_pattern():
    headers_sorted = sorted(SECTION_HEADERS, key=len, reverse=True)
    pattern = "|".join(re.escape(h) for h in headers_sorted)
    return re.compile(rf"^({pattern})(?:\s*\([^)]*\))?:", re.MULTILINE | re.IGNORECASE)


HEADER_PATTERN = _build_header_pattern()


def chunk_note(note_id: str, text: str) -> list[Chunk]:
    """Splits a clinical note into section-based chunks."""
    patient_line = extract_patient_header(text)
    matches = list(HEADER_PATTERN.finditer(text))

    if not matches:
        return [Chunk(note_id, "FULL_NOTE", text.strip(), f"{note_id}_chunk0")]

    chunks = []
    for i, match in enumerate(matches):
        section_name = match.group(1).upper()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        if not section_text:
            continue

        chunk_text = f"Patient context: {patient_line}\n{section_name}:\n{section_text}"
        chunks.append(Chunk(
            note_id=note_id,
            section=section_name,
            text=chunk_text,
            chunk_id=f"{note_id}_{section_name.replace(' ', '_')}"
        ))

    return chunks


def load_and_chunk_all(notes_dir: str) -> list[Chunk]:
    """Loads every .txt note in notes_dir and returns all chunks across all notes."""
    all_chunks = []
    for fname in sorted(os.listdir(notes_dir)):
        if not fname.endswith(".txt"):
            continue
        note_id = os.path.splitext(fname)[0]
        with open(os.path.join(notes_dir, fname), "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_note(note_id, text))
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_all("../data/notes")
    print(f"Total chunks created: {len(chunks)}")
    for c in chunks[:3]:
        print(f"\n--- {c.chunk_id} ---")
        print(c.text[:200], "...")