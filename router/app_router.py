import sys
from pathlib import Path

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Add subfolders
RAG_DIR = BASE_DIR / "rag"
DATA_ANALYSIS_DIR = BASE_DIR / "data_analysis"

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

if str(DATA_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_ANALYSIS_DIR))


# ============================================================
# IMPORTS
# ============================================================

from data_analysis.analyzer import (
    dataset_summary,
    churn_by_subscription,
    churn_by_contract,
    churn_by_gender,
    average_metrics,
    churn_by_payment_delay,
    churn_by_support_calls,
    churn_factor_comparison
)

from data_analysis.ai_analysts import (
    classify_question,
    run_analysis,
    generate_answer
)

from rag.rag_answer import (
    retrieve_documents,
    ask_gemini,
    display_sources
)

from query_router import classify_query

# ============================================================
# DATA ROUTE
# ============================================================

def handle_data_question(question):

    print("\n")
    print("=" * 70)
    print("DATA ANALYSIS")
    print("=" * 70)

    # Ask Gemini which analysis is required
    classification = classify_question(
        question
    )

    analysis_type = classification[
        "analysis_type"
    ]

    print(
        f"\nAnalysis selected: {analysis_type}"
    )

    # Run actual Pandas analysis
    result = run_analysis(
        analysis_type
    )

    # Generate explanation
    answer = generate_answer(
        question,
        analysis_type,
        result
    )

    print("\nAI ANSWER")
    print("-" * 70)

    print(answer)

    print("\nPANDAS RESULT")
    print("-" * 70)

    if hasattr(result, "to_string"):

        print(
            result.to_string(
                index=False
            )
        )

    else:

        print(result)

    return answer


# ============================================================
# RAG ROUTE
# ============================================================

def handle_rag_question(question):

    print("\n")
    print("=" * 70)
    print("RAG ANALYSIS")
    print("=" * 70)

    # Retrieve relevant documents
    results = retrieve_documents(
        question,
        top_k=3
    )

    # Ask Gemini using retrieved context
    answer = ask_gemini(
        question,
        results
    )

    print("\nAI ANSWER")
    print("-" * 70)

    print(answer)

    # Show sources
    display_sources(
        results
    )

    return answer


# ============================================================
# HYBRID ROUTE
# ============================================================

def handle_hybrid_question(question):

    print("\n")
    print("=" * 70)
    print("HYBRID ANALYSIS")
    print("=" * 70)

    print(
        "\nRunning Pandas analysis..."
    )

    # --------------------------------------------------------
    # STEP 1: Determine the DATA analysis
    # --------------------------------------------------------

    classification = classify_question(
        question
    )

    analysis_type = classification[
        "analysis_type"
    ]

    print(
        f"Data analysis selected: "
        f"{analysis_type}"
    )

    # --------------------------------------------------------
    # STEP 2: Run Pandas
    # --------------------------------------------------------

    result = run_analysis(
        analysis_type
    )

    # --------------------------------------------------------
    # STEP 3: Retrieve business documents
    # --------------------------------------------------------

    print(
        "\nSearching business documents..."
    )

    rag_results = retrieve_documents(
        question,
        top_k=3
    )

    # --------------------------------------------------------
    # STEP 4: Prepare Pandas result
    # --------------------------------------------------------

    if hasattr(result, "to_string"):

        data_result = result.to_string(
            index=False
        )

    else:

        data_result = str(result)

    # --------------------------------------------------------
    # STEP 5: Prepare RAG context
    # --------------------------------------------------------

    rag_context_parts = []

    for i, item in enumerate(
        rag_results,
        start=1
    ):

        rag_context_parts.append(
            f"""
SOURCE {i}
Document: {item['source']}
Page: {item['page']}

Content:
{item['text']}
"""
        )

    rag_context = "\n".join(
        rag_context_parts
    )

    # --------------------------------------------------------
    # STEP 6: Combine both sources
    # --------------------------------------------------------

    prompt = f"""
You are ShopSphere's AI Business Analyst.

The user asked:

{question}

You have TWO sources of information.

============================================================
SOURCE 1 — CUSTOMER DATA
============================================================

The following result was calculated directly
from the customer dataset using Pandas:

{data_result}


============================================================
SOURCE 2 — BUSINESS DOCUMENTS
============================================================

The following information was retrieved from
ShopSphere's business documents:

{rag_context}


============================================================
TASK
============================================================

Combine the customer-data analysis and business
document information to answer the user's question.

RULES:

1. Use the Pandas result for numerical/data claims.
2. Use the business documents for policy claims.
3. Do not invent numbers.
4. Do not invent company policies.
5. Clearly distinguish data findings from policy
   recommendations.
6. If information is missing, say so.
7. Give a concise business-focused answer.
8. At the end, list the relevant document names
   and page numbers.

USER QUESTION:

{question}
"""

    # --------------------------------------------------------
    # STEP 7: Gemini
    # --------------------------------------------------------

    from google import genai
    import os

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY was not found."
        )

    client = genai.Client(
        api_key=api_key
    )

    response = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    answer = response.output_text

    # --------------------------------------------------------
    # STEP 8: Display final answer
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("HYBRID AI ANSWER")
    print("=" * 70)

    print(answer)

    # --------------------------------------------------------
    # STEP 9: Display sources
    # --------------------------------------------------------

    display_sources(
        rag_results
    )

    return answer


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    print("\n")
    print("=" * 80)

    print(
        "        SHOPSPHERE AI DATA ANALYST"
    )

    print("=" * 80)

    print(
        "\nAsk questions about your customer data "
        "and business policies."
    )

    print(
        "\nType 'exit' to stop."
    )

    while True:

        question = input(
            "\nYour question: "
        ).strip()

        if question.lower() == "exit":

            print(
                "\nGoodbye!"
            )

            break

        if not question:

            continue

        try:

            # ==================================================
            # QUERY ROUTER
            # ==================================================

            route = classify_query(
                question
            )

            print(
                f"\nRoute selected: {route}"
            )

            # ==================================================
            # DATA
            # ==================================================

            if route == "DATA":

                handle_data_question(
                    question
                )

            # ==================================================
            # RAG
            # ==================================================

            elif route == "RAG":

                handle_rag_question(
                    question
                )

            # ==================================================
            # HYBRID
            # ==================================================

            elif route == "HYBRID":

                handle_hybrid_question(
                    question
                )

            # ==================================================
            # UNKNOWN
            # ==================================================

            else:

                print(
                    "Unknown route returned by Gemini."
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