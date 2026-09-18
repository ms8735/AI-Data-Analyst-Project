import os
import json

from google import genai

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
# 1. CLASSIFY USER QUESTION
# ============================================================

def classify_question(question):

    prompt = f"""
You are an AI Data Analyst.

Classify the user's question into exactly ONE
of these analysis types:

1. dataset_summary
2. churn_by_subscription
3. churn_by_contract
4. churn_by_gender
5. average_metrics
6. churn_by_payment_delay
7. churn_by_support_calls
8. churn_factor_comparison

Return ONLY valid JSON.

Example:

{{
    "analysis_type": "churn_by_subscription"
}}

User question:

{question}
"""

    response = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt
    )

    text = response.output_text.strip()

    text = text.replace(
        "```json",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    text = text.strip()

    return json.loads(text)


# ============================================================
# 2. RUN PANDAS ANALYSIS
# ============================================================

def run_analysis(analysis_type):

    if analysis_type == "dataset_summary":
        return dataset_summary()

    elif analysis_type == "churn_by_subscription":
        return churn_by_subscription()

    elif analysis_type == "churn_by_contract":
        return churn_by_contract()

    elif analysis_type == "churn_by_gender":
        return churn_by_gender()

    elif analysis_type == "average_metrics":
        return average_metrics()

    elif analysis_type == "churn_by_payment_delay":
        return churn_by_payment_delay()

    elif analysis_type == "churn_by_support_calls":
        return churn_by_support_calls()

    elif analysis_type == "churn_factor_comparison":
        return churn_factor_comparison()

    else:
        raise ValueError(
            f"Unknown analysis type: {analysis_type}"
        )


# ============================================================
# 3. CONVERT RESULT TO TEXT
# ============================================================

def format_result(result):

    if hasattr(result, "to_string"):

        return result.to_string(
            index=False
        )

    return str(result)


# ============================================================
# 4. ASK GEMINI TO EXPLAIN RESULT
# ============================================================

def generate_answer(
    question,
    analysis_type,
    result
):

    result_text = format_result(
        result
    )

    prompt = f"""
You are ShopSphere's AI Data Analyst.

The user asked:

{question}

The analysis performed was:

{analysis_type}

The following result was calculated
directly from the customer dataset using Pandas:

{result_text}

Your task is to explain the result clearly.

RULES:

1. Use ONLY the provided analysis result.
2. Do not invent numbers.
3. Do not change any calculated values.
4. Clearly answer the user's question.
5. Mention important patterns or differences.
6. Keep the answer concise but useful.
7. If the question asks "which is highest",
   explicitly identify the highest value.
8. If the question asks "which is lowest",
   explicitly identify the lowest value.
9. Do not say that you calculated something
   that is not present in the result.

Return only the final natural-language answer.
"""

    response = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt
    )

    return response.output_text.strip()


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SHOPSPHERE AI DATA ANALYST")
    print("=" * 70)

    print(
        "\nAsk questions about your customer dataset."
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

            # -----------------------------------------
            # STEP 1: Understand question
            # -----------------------------------------

            classification = classify_question(
                question
            )

            analysis_type = classification[
                "analysis_type"
            ]

            print(
                f"\nAnalysis selected: "
                f"{analysis_type}"
            )

            # -----------------------------------------
            # STEP 2: Perform actual Pandas analysis
            # -----------------------------------------

            result = run_analysis(
                analysis_type
            )

            # -----------------------------------------
            # STEP 3: Generate explanation
            # -----------------------------------------

            answer = generate_answer(
                question,
                analysis_type,
                result
            )

            # -----------------------------------------
            # STEP 4: Display answer
            # -----------------------------------------

            print("\n")
            print("=" * 70)
            print("AI ANSWER")
            print("=" * 70)

            print(answer)

            # -----------------------------------------
            # STEP 5: Display actual calculation
            # -----------------------------------------

            print("\n")
            print("=" * 70)
            print("PANDAS RESULT")
            print("=" * 70)

            print(
                format_result(result)
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