import os
import json

from google import genai


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found."
    )

client = genai.Client(
    api_key=api_key
)

GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# CLASSIFY USER QUERY
# ============================================================

def classify_query(question):

    prompt = f"""
You are the query router for an AI Data Analyst
and business knowledge assistant.

The system has two knowledge sources:

1. DATA
   Customer CSV dataset.
   Used for numerical analysis, statistics,
   comparisons, trends and charts.

2. RAG
   Business policy documents stored in a FAISS
   vector database.
   Used for company policies, rules, procedures,
   terms and business information.

There is also a HYBRID category.

Choose HYBRID when the question requires BOTH:
- information/calculation from the customer dataset
AND
- information from the business documents.

Choose DATA when the answer can be obtained
from the customer dataset.

Choose RAG when the answer can be obtained
from the business documents.

Return ONLY valid JSON.

Format:

{{
    "route": "DATA"
}}

Allowed values:

DATA
RAG
HYBRID

Examples:

Question:
"What is the overall churn rate?"

Answer:
{{
    "route": "DATA"
}}

Question:
"How can a customer cancel their subscription?"

Answer:
{{
    "route": "RAG"
}}

Question:
"Which subscription has the highest churn and
what retention strategy does the company recommend?"

Answer:
{{
    "route": "HYBRID"
}}

User question:

{question}
"""

    response = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt
    )

    text = response.output_text.strip()

    # Remove markdown if Gemini adds it
    text = text.replace(
        "```json",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    text = text.strip()

    result = json.loads(text)

    return result["route"]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SHOPSPHERE QUERY ROUTER")
    print("=" * 70)

    print(
        "\nType 'exit' to stop."
    )

    while True:

        question = input(
            "\nEnter your question: "
        ).strip()

        if question.lower() == "exit":

            print("\nGoodbye!")

            break

        if not question:

            continue

        try:

            route = classify_query(
                question
            )

            print(
                f"\nSelected Route: {route}"
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