import os
import uuid
import pickle
from pathlib import Path

import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "documents"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

FAISS_INDEX_PATH = VECTOR_DB_DIR / "faiss.index"
METADATA_PATH = VECTOR_DB_DIR / "metadata.pkl"


# ============================================================
# RAG CONFIGURATION
# ============================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# 1. EXTRACT TEXT FROM PDF
# ============================================================

def extract_pdf_text(pdf_path):

    pages = []

    document = fitz.open(pdf_path)

    for page_number, page in enumerate(document):

        text = page.get_text("text")

        if text.strip():

            pages.append({
                "source": pdf_path.name,
                "page": page_number + 1,
                "text": text.strip()
            })

    document.close()

    return pages


# ============================================================
# 2. LOAD ALL PDF FILES
# ============================================================

def load_all_documents():

    all_pages = []

    pdf_files = sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR}"
        )

    print(f"\nFound {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:

        print(
            f"Reading: {pdf_file.name}"
        )

        pages = extract_pdf_text(
            pdf_file
        )

        all_pages.extend(pages)

    print(
        f"\nTotal pages extracted: "
        f"{len(all_pages)}"
    )

    return all_pages


# ============================================================
# 3. CREATE TEXT CHUNKS
# ============================================================

def create_chunks(pages):

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0
        chunk_number = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[start:end]

            chunk_text = chunk_text.strip()

            if chunk_text:

                chunks.append({

                    "id": str(uuid.uuid4()),

                    "text": chunk_text,

                    "source": page["source"],

                    "page": page["page"],

                    "chunk_number": chunk_number

                })

            # Move forward while maintaining overlap
            start += CHUNK_SIZE - CHUNK_OVERLAP

            chunk_number += 1

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    return chunks


# ============================================================
# 4. CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "Creating embeddings..."
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    return embeddings


# ============================================================
# 5. CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    print(
        f"\nEmbedding dimension: "
        f"{dimension}"
    )

    # Inner Product works well with
    # normalized embeddings.
    index = faiss.IndexFlatIP(
        dimension
    )

    # Normalize vectors
    faiss.normalize_L2(
        embeddings
    )

    index.add(
        embeddings
    )

    print(
        f"FAISS index created."
    )

    print(
        f"Vectors stored: "
        f"{index.ntotal}"
    )

    return index


# ============================================================
# 6. SAVE FAISS + METADATA
# ============================================================

def save_vector_database(
    index,
    chunks
):

    VECTOR_DB_DIR.mkdir(
        exist_ok=True
    )

    # Save FAISS index
    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH)
    )

    # Save metadata
    with open(
        METADATA_PATH,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print(
        "\nVector database saved."
    )

    print(
        f"FAISS index: "
        f"{FAISS_INDEX_PATH}"
    )

    print(
        f"Metadata: "
        f"{METADATA_PATH}"
    )


# ============================================================
# 7. MAIN FUNCTION
# ============================================================

def main():

    print("=" * 70)

    print(
        "SHOPSPHERE RAG - DOCUMENT INGESTION"
    )

    print("=" * 70)

    # Load PDFs
    pages = load_all_documents()

    # Create chunks
    chunks = create_chunks(
        pages
    )

    # Create embeddings
    embeddings = create_embeddings(
        chunks
    )

    # Create FAISS index
    index = create_faiss_index(
        embeddings
    )

    # Save database
    save_vector_database(
        index,
        chunks
    )

    print("\n" + "=" * 70)

    print(
        "INGESTION COMPLETED SUCCESSFULLY!"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()