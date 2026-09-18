from analyzer import (
    churn_by_subscription,
    churn_by_contract,
    churn_by_gender,
    churn_by_payment_delay,
    churn_by_support_calls
)

from charts import (
    plot_churn_by_subscription,
    plot_churn_by_contract,
    plot_churn_by_gender,
    plot_churn_by_payment_delay,
    plot_churn_by_support_calls
)


print("Creating charts...")


# Subscription
data = churn_by_subscription()

path = plot_churn_by_subscription(data)

print("Subscription chart:", path)


# Contract
data = churn_by_contract()

path = plot_churn_by_contract(data)

print("Contract chart:", path)


# Gender
data = churn_by_gender()

path = plot_churn_by_gender(data)

print("Gender chart:", path)


# Payment Delay
data = churn_by_payment_delay()

path = plot_churn_by_payment_delay(data)

print("Payment delay chart:", path)


# Support Calls
data = churn_by_support_calls()

path = plot_churn_by_support_calls(data)

print("Support calls chart:", path)


print("\nAll charts generated successfully!")