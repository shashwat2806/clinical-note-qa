"""
tests/test_chunker.py
Unit tests for the section-aware clinical note chunker.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from chunker import chunk_note, extract_patient_header, load_and_chunk_all

SAMPLE_NOTE = """DISCHARGE SUMMARY

Patient: Test Patient
Age: 45
Sex: Male
Date of Admission: 2026-01-01
Date of Discharge: 2026-01-05

CHIEF COMPLAINT:
Fever and cough.

PAST MEDICAL HISTORY:
Hypertension.

ALLERGIES:
Penicillin - causes rash.

MEDICATIONS AT DISCHARGE:
- Amoxicillin 500mg twice daily

DISCHARGE DIAGNOSIS:
Community-acquired pneumonia.
"""


class TestExtractPatientHeader(unittest.TestCase):

    def test_extracts_patient_name(self):
        header = extract_patient_header(SAMPLE_NOTE)
        self.assertIn("Test Patient", header)

    def test_extracts_age(self):
        header = extract_patient_header(SAMPLE_NOTE)
        self.assertIn("45", header)

    def test_extracts_sex(self):
        header = extract_patient_header(SAMPLE_NOTE)
        self.assertIn("Male", header)

    def test_uses_pipe_separator(self):
        header = extract_patient_header(SAMPLE_NOTE)
        self.assertIn("|", header)

    def test_empty_text_returns_empty(self):
        header = extract_patient_header("")
        self.assertEqual(header, "")


class TestChunkNote(unittest.TestCase):

    def setUp(self):
        self.chunks = chunk_note("test_note", SAMPLE_NOTE)

    def test_produces_chunks(self):
        self.assertGreater(len(self.chunks), 0)

    def test_chunk_ids_are_unique(self):
        ids = [c.chunk_id for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_note_id_is_set(self):
        for chunk in self.chunks:
            self.assertEqual(chunk.note_id, "test_note")

    def test_allergies_section_found(self):
        sections = [c.section for c in self.chunks]
        self.assertIn("ALLERGIES", sections)

    def test_discharge_diagnosis_found(self):
        sections = [c.section for c in self.chunks]
        self.assertIn("DISCHARGE DIAGNOSIS", sections)

    def test_patient_context_in_chunk_text(self):
        for chunk in self.chunks:
            self.assertIn("Test Patient", chunk.text)

    def test_empty_note_returns_one_chunk(self):
        chunks = chunk_note("empty", "")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section, "FULL_NOTE")


class TestLoadAndChunkAll(unittest.TestCase):

    def test_loads_all_notes(self):
        notes_dir = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'notes'
        )
        chunks = load_and_chunk_all(notes_dir)
        self.assertGreater(len(chunks), 100)

    def test_all_chunks_have_note_id(self):
        notes_dir = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'notes'
        )
        chunks = load_and_chunk_all(notes_dir)
        for chunk in chunks:
            self.assertTrue(chunk.note_id.startswith("note_"))


if __name__ == "__main__":
    unittest.main(verbosity=2)