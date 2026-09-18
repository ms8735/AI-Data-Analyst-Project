import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHARTS_DIR = BASE_DIR / "charts"

CHARTS_DIR.mkdir(exist_ok=True)


# ============================================================
# CHURN BY SUBSCRIPTION
# ============================================================

def plot_churn_by_subscription(data):

    plt.figure(figsize=(8, 5))

    plt.bar(
        data["Subscription Type"],
        data["churn_rate"]
    )

    plt.xlabel("Subscription Type")
    plt.ylabel("Churn Rate (%)")
    plt.title("Churn Rate by Subscription Type")

    plt.xticks(rotation=30)

    plt.tight_layout()

    path = CHARTS_DIR / "churn_by_subscription.png"

    plt.savefig(path)

    plt.close()

    return str(path)


# ============================================================
# CHURN BY CONTRACT
# ============================================================

def plot_churn_by_contract(data):

    plt.figure(figsize=(8, 5))

    plt.bar(
        data["Contract Length"],
        data["churn_rate"]
    )

    plt.xlabel("Contract Length")
    plt.ylabel("Churn Rate (%)")
    plt.title("Churn Rate by Contract Length")

    plt.xticks(rotation=30)

    plt.tight_layout()

    path = CHARTS_DIR / "churn_by_contract.png"

    plt.savefig(path)

    plt.close()

    return str(path)


# ============================================================
# CHURN BY GENDER
# ============================================================

def plot_churn_by_gender(data):

    plt.figure(figsize=(7, 5))

    plt.bar(
        data["Gender"],
        data["churn_rate"]
    )

    plt.xlabel("Gender")
    plt.ylabel("Churn Rate (%)")
    plt.title("Churn Rate by Gender")

    plt.tight_layout()

    path = CHARTS_DIR / "churn_by_gender.png"

    plt.savefig(path)

    plt.close()

    return str(path)


# ============================================================
# CHURN BY PAYMENT DELAY
# ============================================================

def plot_churn_by_payment_delay(data):

    plt.figure(figsize=(9, 5))

    plt.plot(
        data["Payment Delay"],
        data["Churn Rate"],
        marker="o"
    )

    plt.xlabel("Payment Delay")
    plt.ylabel("Churn Rate (%)")
    plt.title("Churn Rate vs Payment Delay")

    plt.tight_layout()

    path = CHARTS_DIR / "churn_by_payment_delay.png"

    plt.savefig(path)

    plt.close()

    return str(path)


# ============================================================
# CHURN BY SUPPORT CALLS
# ============================================================

def plot_churn_by_support_calls(data):

    plt.figure(figsize=(9, 5))

    plt.plot(
        data["Support Calls"],
        data["Churn Rate"],
        marker="o"
    )

    plt.xlabel("Support Calls")
    plt.ylabel("Churn Rate (%)")
    plt.title("Churn Rate vs Support Calls")

    plt.tight_layout()

    path = CHARTS_DIR / "churn_by_support_calls.png"

    plt.savefig(path)

    plt.close()

    return str(path)