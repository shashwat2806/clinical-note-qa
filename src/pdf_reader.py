"""
pdf_reader.py
Extracts text from an uploaded PDF clinical note.
The extracted text feeds directly into the existing chunker —
no changes to the RAG pipeline needed.
"""

import pdfplumber


def extract_text_from_pdf(pdf_file) -> str:
    """
    Takes a PDF file object (from Streamlit's file_uploader)
    and returns all text as a single string.

    Args:
        pdf_file: file-like object (BytesIO) from st.file_uploader

    Returns:
        str: full extracted text from all pages, or empty string if extraction fails
    """
    page_texts = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:  # skip blank or image-only pages
                page_texts.append(text)

    return "\n".join(page_texts)


def is_valid_pdf_text(text: str, min_chars: int = 100) -> bool:
    """
    Basic sanity check — returns False if the extracted text is
    too short to be a real clinical note (likely a scanned image PDF
    where pdfplumber can't extract text).
    """
    return len(text.strip()) >= min_chars


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        with open(pdf_path, "rb") as f:
            text = extract_text_from_pdf(f)
        print(f"Extracted {len(text)} characters")
        print(text[:500], "...")
    else:
        print("pdf_reader.py loaded successfully")
        print("Usage: python pdf_reader.py path/to/file.pdf")