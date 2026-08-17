import json
import re
from pathlib import Path

import pymupdf


# ============================================================
# DIRECTORIES
# ============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DOCUMENT METADATA
# ============================================================

CATEGORY_INFO = {
    "central": {
        "jurisdiction": "India",
        "document_type": "Central Law / Regulation"
    },

    "traffic": {
        "jurisdiction": "India",
        "document_type": "Traffic / Road Safety"
    },

    "state": {
        "jurisdiction": "Maharashtra",
        "document_type": "State Motor Vehicle Rules"
    },

    "supplementary": {
        "jurisdiction": "India",
        "document_type": "Supplementary Road Safety"
    }
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving legal content.
    """

    # Normalize line breaks
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text
    )

    return text.strip()


# ============================================================
# DOCUMENT METADATA
# ============================================================

def get_document_metadata(pdf_path: Path):
    """
    Determine category, jurisdiction and document type
    from the folder in which the PDF exists.
    """

    try:
        relative_path = pdf_path.relative_to(RAW_DIR)

        # First folder after data/raw/
        parts = relative_path.parts

        if len(parts) >= 2:
            category = parts[0].lower()
        else:
            category = "unknown"

    except ValueError:
        category = "unknown"

    metadata = CATEGORY_INFO.get(
        category,
        {
            "jurisdiction": "Unknown",
            "document_type": "Unknown"
        }
    )

    return {
        "category": category,
        "jurisdiction": metadata["jurisdiction"],
        "document_type": metadata["document_type"]
    }


# ============================================================
# EXTRACT PDF
# ============================================================

def extract_pdf(pdf_path: Path):
    """
    Extract text page-by-page.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text")

        text = clean_text(text)

        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })

    document.close()

    return pages


# ============================================================
# SECTION DETECTION
# ============================================================

def split_into_sections(text: str):
    """
    Attempt to split legal text at major numbered sections.

    Examples:

        8. Grant of learner's licence
        9. Grant of driving licence
        10. ...
    """

    pattern = r"(?=\b\d+\.\s+[A-Z])"

    sections = re.split(
        pattern,
        text
    )

    sections = [
        section.strip()
        for section in sections
        if section.strip()
    ]

    return sections


# ============================================================
# CHUNKING
# ============================================================

def chunk_section(
    text: str,
    max_words: int = 450,
    overlap: int = 80
):
    """
    Split large sections into overlapping chunks.
    """

    words = text.split()

    if len(words) <= max_words:
        return [text]

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + max_words,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(
                chunk.strip()
            )

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# ============================================================
# SECTION NUMBER
# ============================================================

def detect_section(text: str):
    """
    Extract section number from the beginning of a chunk.
    """

    # Standard:
    # 129. Wearing of protective headgear
    match = re.match(
        r"^(\d+)\.\s+",
        text
    )

    if match:
        return match.group(1)

    # Alternative format:
    # Section 129
    match = re.match(
        r"^Section\s+(\d+)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return "unknown"


# ============================================================
# SECTION TITLE
# ============================================================

def detect_section_title(text: str):
    """
    Try to extract the title from a legal section.

    Example:

        129. Wearing of protective headgear.—Every person...

    Returns:

        Wearing of protective headgear
    """

    match = re.match(
        r"^\d+\.\s+(.+?)(?:—|-{1,2}|\.—|\.)\s+",
        text
    )

    if match:

        title = match.group(1).strip()

        if len(title) <= 200:
            return title

    return None


# ============================================================
# PROCESS SINGLE PDF
# ============================================================

def process_pdf(
    pdf_path: Path,
    starting_id: int
):

    print(
        f"\nProcessing: {pdf_path.name}"
    )

    metadata = get_document_metadata(
        pdf_path
    )

    print(
        f"Category    : {metadata['category']}"
    )

    print(
        f"Jurisdiction: {metadata['jurisdiction']}"
    )

    print(
        f"Type        : {metadata['document_type']}"
    )

    pages = extract_pdf(
        pdf_path
    )

    print(
        f"Pages with text: {len(pages)}"
    )

    all_chunks = []

    chunk_id = starting_id

    for page in pages:

        sections = split_into_sections(
            page["text"]
        )

        for section_text in sections:

            chunks = chunk_section(
                section_text
            )

            for chunk in chunks:

                section_number = detect_section(
                    chunk
                )

                section_title = detect_section_title(
                    chunk
                )

                all_chunks.append({

                    "id": chunk_id,

                    "text": chunk,

                    "source": pdf_path.name,

                    "source_path": str(
                        pdf_path.relative_to(RAW_DIR)
                    ).replace("\\", "/"),

                    "page": page["page"],

                    "section": section_number,

                    "section_title": section_title,

                    "category": metadata["category"],

                    "jurisdiction": metadata["jurisdiction"],

                    "document_type": metadata["document_type"]

                })

                chunk_id += 1

    return all_chunks, chunk_id


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ROADsOS LEGAL DOCUMENT INGESTION")
    print("=" * 60)

    if not RAW_DIR.exists():

        print(
            f"ERROR: {RAW_DIR} does not exist."
        )

        return

    # --------------------------------------------------------
    # IMPORTANT:
    # rglob searches recursively inside all subfolders.
    # --------------------------------------------------------

    pdf_files = sorted(
        RAW_DIR.rglob("*.pdf")
    )

    if not pdf_files:

        print(
            "\nNo PDF files found."
        )

        print(
            f"Expected PDFs inside: {RAW_DIR}"
        )

        return

    print(
        f"\nFound {len(pdf_files)} PDF documents:\n"
    )

    for pdf in pdf_files:

        print(
            f"  - {pdf.relative_to(RAW_DIR)}"
        )

    print()

    # --------------------------------------------------------
    # PROCESS ALL DOCUMENTS
    # --------------------------------------------------------

    all_chunks = []

    next_id = 0

    for pdf_path in pdf_files:

        chunks, next_id = process_pdf(
            pdf_path,
            next_id
        )

        all_chunks.extend(
            chunks
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = (
        PROCESSED_DIR /
        "chunks.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LEGAL DOCUMENT PROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Total documents : {len(pdf_files)}"
    )

    print(
        f"Total chunks    : {len(all_chunks)}"
    )

    print(
        f"Output          : {output_path}"
    )

    # --------------------------------------------------------
    # CATEGORY SUMMARY
    # --------------------------------------------------------

    category_counts = {}

    for chunk in all_chunks:

        category = chunk["category"]

        category_counts[category] = (
            category_counts.get(
                category,
                0
            ) + 1
        )

    print("\nChunks by category:")

    for category, count in category_counts.items():

        print(
            f"  {category}: {count}"
        )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()