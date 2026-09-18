import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DB_DIR = BASE_DIR / "vector_db"

FAISS_INDEX_PATH = VECTOR_DB_DIR / "faiss.index"
METADATA_PATH = VECTOR_DB_DIR / "metadata.pkl"


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 3


# ============================================================
# LOAD FAISS
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(
    str(FAISS_INDEX_PATH)
)

print(
    f"Vectors in index: "
    f"{index.ntotal}"
)


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    METADATA_PATH,
    "rb"
) as file:

    metadata = pickle.load(
        file
    )


print(
    f"Metadata records: "
    f"{len(metadata)}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "Embedding model loaded."
)


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(
    query,
    top_k=TOP_K
):

    # Create query embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    # Normalize query vector
    faiss.normalize_L2(
        query_embedding
    )

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        result = metadata[idx].copy()

        result["score"] = float(
            score
        )

        results.append(
            result
        )

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results
):

    print("\n")
    print("=" * 80)

    print(
        "QUESTION:"
    )

    print(query)

    print("=" * 80)

    for i, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"RESULT {i}"
        )

        print("-" * 80)

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Chunk: "
            f"{result['chunk_number']}"
        )

        print(
            f"Similarity score: "
            f"{result['score']:.4f}"
        )

        print(
            "\nText:"
        )

        print(
            result["text"]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)

    print(
        "SHOPSPHERE RAG RETRIEVAL SYSTEM"
    )

    print("=" * 80)

    print(
        "\nType 'exit' to stop."
    )

    while True:

        query = input(
            "\nAsk your question: "
        ).strip()

        if query.lower() == "exit":

            print(
                "\nGoodbye!"
            )

            break

        if not query:

            print(
                "Please enter a question."
            )

            continue

        results = retrieve_documents(
            query
        )

        display_results(
            query,
            results
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()