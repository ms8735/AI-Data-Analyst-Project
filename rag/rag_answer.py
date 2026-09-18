import os
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# PATHS
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

# Current Gemini model
GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# GEMINI CLIENT
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found.\n"
        "Please set your Gemini API key as an environment variable."
    )

client = genai.Client(
    api_key=api_key
)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(
    str(FAISS_INDEX_PATH)
)

print(
    f"FAISS vectors loaded: {index.ntotal}"
)


# ============================================================
# LOAD METADATA
# ============================================================

print("Loading metadata...")

with open(
    METADATA_PATH,
    "rb"
) as file:

    metadata = pickle.load(file)

print(
    f"Metadata records loaded: {len(metadata)}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    top_k=TOP_K
):

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

        result["score"] = float(score)

        results.append(result)

    return results


# ============================================================
# CREATE CONTEXT
# ============================================================

def create_context(results):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}
Document: {result['source']}
Page: {result['page']}

Content:
{result['text']}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# ASK GEMINI
# ============================================================

def ask_gemini(
    question,
    results
):

    context = create_context(
        results
    )

    prompt = f"""
You are ShopSphere's AI business assistant.

Answer the user's question using ONLY the
information provided in the context.

RULES:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the
   context, say:
   "I could not find this information
   in the knowledge base."
4. Give a clear and concise answer.
5. Mention the relevant document and
   page number at the end.

CONTEXT:
{context}

USER QUESTION:
{question}
"""

    # Current Gemini Interactions API
    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt
    )

    return interaction.output_text


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    print("\n")
    print("=" * 80)
    print("RETRIEVED SOURCES")
    print("=" * 80)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{i}. "
            f"{result['source']} "
            f"(Page {result['page']}) "
            f"| Similarity: {result['score']:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("SHOPSPHERE RAG ASSISTANT")
    print("=" * 80)

    print(
        "\nAsk questions about ShopSphere's policies."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        question = input(
            "\nYour question: "
        ).strip()

        if question.lower() == "exit":

            print("\nGoodbye!")

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        try:

            # ------------------------------------
            # STEP 1: Retrieve from FAISS
            # ------------------------------------

            results = retrieve_documents(
                question,
                TOP_K
            )

            # ------------------------------------
            # STEP 2: Send retrieved context
            # to Gemini
            # ------------------------------------

            answer = ask_gemini(
                question,
                results
            )

            # ------------------------------------
            # STEP 3: Display answer
            # ------------------------------------

            print("\n")
            print("=" * 80)
            print("ANSWER")
            print("=" * 80)

            print(answer)

            # ------------------------------------
            # STEP 4: Display sources
            # ------------------------------------

            display_sources(
                results
            )

        except Exception as e:

            print(
                f"\nError: {e}"
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()