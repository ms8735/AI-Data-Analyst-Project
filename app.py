import streamlit as st
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR))


# ============================================================
# IMPORT BACKEND FUNCTIONS
# ============================================================

from router.query_router import classify_query

from data_analysis.ai_analysts import (
    classify_question,
    run_analysis,
    generate_answer
)

from rag.rag_answer import (
    retrieve_documents,
    ask_gemini
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ShopSphere AI Analyst",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 ShopSphere AI Data Analyst")

st.markdown(
    """
### AI-powered Customer Analytics & Business Intelligence

Ask questions about:

- 📈 Customer data
- 🔍 Churn analysis
- 📚 Business policies
- 🤖 Business documents
- ⭐ Hybrid data + document analysis
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔀 Analysis Modes")

    st.markdown(
        """
        **DATA**

        Uses Pandas to analyze the customer dataset.

        **RAG**

        Searches business documents using FAISS.

        **HYBRID**

        Combines customer data + business documents.
        """
    )

    st.divider()

    st.header("💡 Example Questions")

    st.markdown(
        """
        **Data**

        • What is the overall churn rate?

        • Which subscription has the highest churn?

        **RAG**

        • How can a customer cancel?

        • What is the payment policy?

        **Hybrid**

        • Which subscription has the highest churn
          and what retention strategy is recommended?
        """
    )


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(
    "💬 Ask your question",
    placeholder="Example: Which subscription has the highest churn?",
    height=100
)


analyze_button = st.button(
    "🔍 Analyze",
    type="primary"
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    try:

        # ====================================================
        # STEP 1 — ROUTER
        # ====================================================

        with st.spinner(
            "Understanding your question..."
        ):

            route = classify_query(
                question
            )


        # ====================================================
        # SHOW ROUTE
        # ====================================================

        st.subheader("🔀 Analysis Route")

        if route == "DATA":

            st.info(
                "📊 DATA — Analyzing customer dataset"
            )

        elif route == "RAG":

            st.info(
                "📚 RAG — Searching business documents"
            )

        elif route == "HYBRID":

            st.info(
                "⭐ HYBRID — Combining data + documents"
            )


        # ====================================================
        # DATA ROUTE
        # ====================================================

        if route == "DATA":

            with st.spinner(
                "Analyzing customer data..."
            ):

                classification = classify_question(
                    question
                )

                analysis_type = classification[
                    "analysis_type"
                ]

                result = run_analysis(
                    analysis_type
                )

                answer = generate_answer(
                    question,
                    analysis_type,
                    result
                )


            st.subheader(
                "🤖 AI Answer"
            )

            st.write(answer)


            st.subheader(
                "📊 Analysis Result"
            )


            if hasattr(
                result,
                "to_string"
            ):

                st.dataframe(
                    result,
                    use_container_width=True
                )

            else:

                st.json(result)


        # ====================================================
        # RAG ROUTE
        # ====================================================

        elif route == "RAG":

            with st.spinner(
                "Searching business documents..."
            ):

                results = retrieve_documents(
                    question,
                    top_k=3
                )

                answer = ask_gemini(
                    question,
                    results
                )


            st.subheader(
                "🤖 AI Answer"
            )

            st.write(answer)


            # ================================================
            # SOURCES
            # ================================================

            st.subheader(
                "📚 Sources"
            )


            for i, item in enumerate(
                results,
                start=1
            ):

                with st.expander(
                    f"Source {i}: {item['source']} — Page {item['page']}"
                ):

                    st.write(
                        item["text"]
                    )


        # ====================================================
        # HYBRID ROUTE
        # ====================================================

        elif route == "HYBRID":

            with st.spinner(
                "Combining customer data and business documents..."
            ):

                # --------------------------------------------
                # DATA
                # --------------------------------------------

                classification = classify_question(
                    question
                )

                analysis_type = classification[
                    "analysis_type"
                ]

                data_result = run_analysis(
                    analysis_type
                )


                # --------------------------------------------
                # RAG
                # --------------------------------------------

                rag_results = retrieve_documents(
                    question,
                    top_k=3
                )


                # --------------------------------------------
                # FORMAT DATA
                # --------------------------------------------

                if hasattr(
                    data_result,
                    "to_string"
                ):

                    data_text = data_result.to_string(
                        index=False
                    )

                else:

                    data_text = str(
                        data_result
                    )


                # --------------------------------------------
                # FORMAT DOCUMENTS
                # --------------------------------------------

                document_context = ""

                for i, item in enumerate(
                    rag_results,
                    start=1
                ):

                    document_context += f"""

SOURCE {i}

Document:
{item['source']}

Page:
{item['page']}

Content:
{item['text']}

"""


                # --------------------------------------------
                # GEMINI
                # --------------------------------------------

                import os
                from google import genai


                api_key = os.getenv(
                    "GEMINI_API_KEY"
                )


                client = genai.Client(
                    api_key=api_key
                )


                prompt = f"""

You are ShopSphere's AI Business Analyst.

USER QUESTION:

{question}


============================================================
CUSTOMER DATA ANALYSIS
============================================================

{data_text}


============================================================
BUSINESS DOCUMENTS
============================================================

{document_context}


============================================================
INSTRUCTIONS
============================================================

Answer the user's question using BOTH sources.

Use customer data for numerical findings.

Use business documents for company policies,
recommendations and procedures.

Do not invent numbers.

Do not invent policies.

Clearly explain the important business insight.

Mention document sources and page numbers
when using policy information.

"""


                response = client.interactions.create(
                    model="gemini-3.6-flash",
                    input=prompt
                )


                answer = response.output_text


            # ================================================
            # FINAL ANSWER
            # ================================================

            st.subheader(
                "⭐ Hybrid AI Answer"
            )

            st.write(answer)


            # ================================================
            # DATA RESULT
            # ================================================

            st.subheader(
                "📊 Customer Data Result"
            )


            if hasattr(
                data_result,
                "to_string"
            ):

                st.dataframe(
                    data_result,
                    use_container_width=True
                )

            else:

                st.json(
                    data_result
                )


            # ================================================
            # SOURCES
            # ================================================

            st.subheader(
                "📚 Business Document Sources"
            )


            for i, item in enumerate(
                rag_results,
                start=1
            ):

                with st.expander(
                    f"{item['source']} — Page {item['page']}"
                ):

                    st.write(
                        item["text"]
                    )


    except Exception as e:

        st.error(
            f"Something went wrong: {e}"
        )