
import re
from pypdf import PdfReader
from text_splitter import split_text


def detect_heading(page_text):
    """
    Attempt to identify a heading from the extracted page text.
    This is a heuristic, not guaranteed PDF layout detection.
    """

    for line in page_text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        words = line.split()

        # Skip very long lines and ordinary full sentences.
        if len(words) > 12:
            continue

        if line.endswith((".", "?", ",")):
            continue

        # Short, uppercase lines are often section headings.
        if line.isupper() and len(words) >= 2:
            return line

        # Also recognize common numbered heading formats:
        # 1 Introduction, 2.1 Architecture, Module - I
        if re.match(
            r"^(?:\d+(?:\.\d+)*[.)]?\s+\S|module\s*[-:]?\s*[ivx0-9]+)",
            line,
            re.IGNORECASE
        ):
            return line

    return "Unknown section"


def process_pdf(file_path, document_name):
    """
    Extract text page by page and attach metadata
    to every chunk.
    """

    reader = PdfReader(file_path)
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if not page_text or not page_text.strip():
            continue

        page_text = page_text.strip()
        heading = detect_heading(page_text)

        page_chunks = split_text(
            page_text,
            chunk_size=350,
            overlap=60
        )

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1
        ):
            chunks.append({
                "text": chunk_text,
                "source": document_name,
                "page": page_number,
                "chunk_id": f"{document_name}-p{page_number}-c{chunk_number}",
                "heading": heading
            })

    return chunks, len(reader.pages)
