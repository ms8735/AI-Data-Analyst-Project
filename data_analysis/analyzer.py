import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "customer_churn_dataset-testing-master.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading customer dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded successfully: "
    f"{df.shape[0]:,} rows × {df.shape[1]} columns"
)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

def dataset_summary():

    return {
        "total_customers": len(df),

        "total_columns": len(df.columns),

        "churned_customers": int(
            df["Churn"].sum()
        ),

        "retained_customers": int(
            (df["Churn"] == 0).sum()
        ),

        "overall_churn_rate": round(
            df["Churn"].mean() * 100,
            2
        )
    }


# ============================================================
# CHURN BY SUBSCRIPTION TYPE
# ============================================================

def churn_by_subscription():

    result = (
        df.groupby("Subscription Type")["Churn"]
        .agg(
            customers="count",
            churned="sum",
            churn_rate="mean"
        )
        .reset_index()
    )

    result["churn_rate"] = (
        result["churn_rate"] * 100
    ).round(2)

    return result.sort_values(
        "churn_rate",
        ascending=False
    )


# ============================================================
# CHURN BY CONTRACT LENGTH
# ============================================================

def churn_by_contract():

    result = (
        df.groupby("Contract Length")["Churn"]
        .agg(
            customers="count",
            churned="sum",
            churn_rate="mean"
        )
        .reset_index()
    )

    result["churn_rate"] = (
        result["churn_rate"] * 100
    ).round(2)

    return result.sort_values(
        "churn_rate",
        ascending=False
    )


# ============================================================
# CHURN BY GENDER
# ============================================================

def churn_by_gender():

    result = (
        df.groupby("Gender")["Churn"]
        .agg(
            customers="count",
            churned="sum",
            churn_rate="mean"
        )
        .reset_index()
    )

    result["churn_rate"] = (
        result["churn_rate"] * 100
    ).round(2)

    return result


# ============================================================
# AVERAGE CUSTOMER METRICS
# ============================================================

def average_metrics():

    return {
        "average_age": round(
            df["Age"].mean(),
            2
        ),

        "average_tenure": round(
            df["Tenure"].mean(),
            2
        ),

        "average_usage_frequency": round(
            df["Usage Frequency"].mean(),
            2
        ),

        "average_support_calls": round(
            df["Support Calls"].mean(),
            2
        ),

        "average_payment_delay": round(
            df["Payment Delay"].mean(),
            2
        ),

        "average_total_spend": round(
            df["Total Spend"].mean(),
            2
        ),

        "average_last_interaction": round(
            df["Last Interaction"].mean(),
            2
        )
    }


# ============================================================
# CHURN BY PAYMENT DELAY
# ============================================================

def churn_by_payment_delay():

    result = (
        df.groupby("Payment Delay")["Churn"]
        .mean()
        .reset_index()
    )

    result["Churn Rate"] = (
        result["Churn"] * 100
    ).round(2)

    result.drop(
        columns=["Churn"],
        inplace=True
    )

    return result


# ============================================================
# CHURN BY SUPPORT CALLS
# ============================================================

def churn_by_support_calls():

    result = (
        df.groupby("Support Calls")["Churn"]
        .mean()
        .reset_index()
    )

    result["Churn Rate"] = (
        result["Churn"] * 100
    ).round(2)

    result.drop(
        columns=["Churn"],
        inplace=True
    )

    return result


# ============================================================
# TOP CHURN FACTORS
# ============================================================

def churn_factor_comparison():

    churned = df[df["Churn"] == 1]

    retained = df[df["Churn"] == 0]

    result = pd.DataFrame({
        "Metric": [
            "Average Age",
            "Average Tenure",
            "Average Usage Frequency",
            "Average Support Calls",
            "Average Payment Delay",
            "Average Total Spend",
            "Average Last Interaction"
        ],

        "Churned Customers": [
            churned["Age"].mean(),
            churned["Tenure"].mean(),
            churned["Usage Frequency"].mean(),
            churned["Support Calls"].mean(),
            churned["Payment Delay"].mean(),
            churned["Total Spend"].mean(),
            churned["Last Interaction"].mean()
        ],

        "Retained Customers": [
            retained["Age"].mean(),
            retained["Tenure"].mean(),
            retained["Usage Frequency"].mean(),
            retained["Support Calls"].mean(),
            retained["Payment Delay"].mean(),
            retained["Total Spend"].mean(),
            retained["Last Interaction"].mean()
        ]
    })

    result["Churned Customers"] = (
        result["Churned Customers"].round(2)
    )

    result["Retained Customers"] = (
        result["Retained Customers"].round(2)
    )

    return result


# ============================================================
# TEST FUNCTIONS
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("CUSTOMER CHURN ANALYSIS")
    print("=" * 70)

    print("\nDATASET SUMMARY")
    print(dataset_summary())

    print("\nCHURN BY SUBSCRIPTION")
    print(churn_by_subscription().to_string(index=False))

    print("\nCHURN BY CONTRACT")
    print(churn_by_contract().to_string(index=False))

    print("\nCHURN BY GENDER")
    print(churn_by_gender().to_string(index=False))

    print("\nAVERAGE METRICS")
    print(average_metrics())

    print("\nCHURN FACTOR COMPARISON")
    print(
        churn_factor_comparison()
        .to_string(index=False)
    )