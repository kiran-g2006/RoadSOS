import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

CHUNKS_FILE = Path(
    "data/processed/chunks.json"
)

VECTORSTORE_DIR = Path(
    "vectorstore"
)

VECTORSTORE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


INDEX_FILE = (
    VECTORSTORE_DIR /
    "road_safety.index"
)

METADATA_FILE = (
    VECTORSTORE_DIR /
    "metadata.json"
)


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    if not CHUNKS_FILE.exists():

        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    if not chunks:

        raise ValueError(
            "chunks.json is empty."
        )

    return chunks


# ============================================================
# VALIDATE CHUNKS
# ============================================================

def validate_chunks(chunks):

    required_fields = [
        "id",
        "text",
        "source",
        "page",
        "section",
        "category",
        "jurisdiction",
        "document_type"
    ]

    valid_chunks = []

    for chunk in chunks:

        missing = [
            field
            for field in required_fields
            if field not in chunk
        ]

        if missing:

            print(
                f"Warning: Chunk {chunk.get('id')} "
                f"is missing fields: {missing}"
            )

        if not chunk.get("text", "").strip():

            continue

        valid_chunks.append(chunk)

    return valid_chunks


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def main():

    print("=" * 60)
    print("BUILDING ROADsOS VECTOR DATABASE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    print(
        f"Loaded chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Validate chunks
    # --------------------------------------------------------

    chunks = validate_chunks(
        chunks
    )

    print(
        f"Valid chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print(
        "\nGenerating embeddings..."
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Ensure float32
    # --------------------------------------------------------

    embeddings = embeddings.astype(
        np.float32
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    # --------------------------------------------------------
    # Add embeddings
    # --------------------------------------------------------

    index.add(
        embeddings
    )

    print(
        f"FAISS vectors: {index.ntotal}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if index.ntotal != len(chunks):

        raise RuntimeError(
            "FAISS vector count does not match "
            "metadata count."
        )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    category_counts = {}

    for chunk in chunks:

        category = chunk.get(
            "category",
            "unknown"
        )

        category_counts[category] = (
            category_counts.get(
                category,
                0
            ) + 1
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VECTOR DATABASE CREATED")
    print("=" * 60)

    print(
        f"Vectors : {index.ntotal}"
    )

    print(
        f"Dimension: {dimension}"
    )

    print(
        f"Index   : {INDEX_FILE}"
    )

    print(
        f"Metadata: {METADATA_FILE}"
    )

    print(
        "\nChunks by category:"
    )

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