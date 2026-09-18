import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DB_DIR = BASE_DIR / "vector_db"

COLLECTION_NAME = "shopsphere_policies"

# Number of relevant chunks to retrieve
TOP_K = 3


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"Connected successfully. "
    f"Total chunks: {collection.count()}"
)


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(query, top_k=TOP_K):

    # Convert user question into embedding
    query_embedding = embedding_model.encode(
        query
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_documents = []

    for i in range(len(results["documents"][0])):

        document = {
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "chunk_number": results["metadatas"][0][i]["chunk_number"],
            "distance": results["distances"][0][i]
        }

        retrieved_documents.append(document)

    return retrieved_documents


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(query, documents):

    print("\n")
    print("=" * 80)
    print("USER QUESTION")
    print("=" * 80)

    print(query)

    for i, document in enumerate(documents, start=1):

        print("\n")
        print("-" * 80)
        print(f"RESULT {i}")
        print("-" * 80)

        print(
            f"Source       : {document['source']}"
        )

        print(
            f"Page         : {document['page']}"
        )

        print(
            f"Chunk        : {document['chunk_number']}"
        )

        print(
            f"Distance     : {document['distance']:.4f}"
        )

        print("\nRetrieved Text:")

        print(document["text"])


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("SHOPSPHERE RAG RETRIEVAL SYSTEM")
    print("=" * 80)

    print("\nType 'exit' to stop.")

    while True:

        query = input("\nAsk your question: ").strip()

        if query.lower() == "exit":
            print("\nExiting...")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:

            documents = retrieve_documents(
                query,
                TOP_K
            )

            display_results(
                query,
                documents
            )

        except Exception as e:

            print(
                f"\nError while retrieving documents: {e}"
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()