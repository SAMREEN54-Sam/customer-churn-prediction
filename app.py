import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Page title
# -----------------------------
st.title("Customer Churn Prediction")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/customer_churn.csv")

# -----------------------------
# Prepare data
# -----------------------------
df = df.dropna()

# Encode Contract
contract_encoder = LabelEncoder()
df["Contract"] = contract_encoder.fit_transform(df["Contract"])

# Encode Churn
churn_encoder = LabelEncoder()
df["Churn"] = churn_encoder.fit_transform(df["Churn"])

# Features and target
X = df[["Age", "MonthlyCharges", "Tenure", "Contract"]]
y = df["Churn"]

# -----------------------------
# Train model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.success("Model is ready for prediction!")

# -----------------------------
# User input
# -----------------------------
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=100,
    value=12
)

contract = st.selectbox(
    "Contract",
    contract_encoder.classes_
)

# Convert contract to number
contract_value = contract_encoder.transform([contract])[0]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Test Churn Prediction"):

    input_data = pd.DataFrame({
        "Age": [age],
        "MonthlyCharges": [monthly_charges],
        "Tenure": [tenure],
        "Contract": [contract_value]
    })

    prediction = model.predict(input_data)[0]

    # Convert prediction back to original Churn value
    result = churn_encoder.inverse_transform([prediction])[0]

    if str(result).lower() == "yes":
        st.error("Prediction: Customer may Churn")
    else:
        st.success("Prediction: Customer is likely to Stay")


        # -----------------------------
# Business Analytics Dashboard
# -----------------------------

st.header("Customer Churn Dashboard")

# Total customers
total_customers = len(df)

# Churn count
churn_count = df["Churn"].sum()

# Stayed count
stay_count = total_customers - churn_count

# Display metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", total_customers)

with col2:
    st.metric("Customers Churned", churn_count)

with col3:
    st.metric("Customers Stayed", stay_count)

# -----------------------------
# Churn Distribution
# -----------------------------

st.subheader("Churn Distribution")

churn_chart = df["Churn"].value_counts()

st.bar_chart(churn_chart)

# -----------------------------
# Contract Distribution
# -----------------------------

st.subheader("Contract Distribution")

contract_chart = df["Contract"].value_counts()

st.bar_chart(contract_chart)

# -----------------------------
# Average Monthly Charges
# -----------------------------

st.subheader("Average Monthly Charges")

average_charges = df["MonthlyCharges"].mean()

st.write(
    f"Average Monthly Charges: {average_charges:.2f}"
)

# -----------------------------
# Average Tenure
# -----------------------------

st.subheader("Average Customer Tenure")

average_tenure = df["Tenure"].mean()

st.write(
    f"Average Tenure: {average_tenure:.2f} months"
)
# -----------------------------
# Churn Rate
# -----------------------------

churn_rate = (churn_count / total_customers) * 100

st.subheader("Overall Churn Rate")

st.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)
# -----------------------------
# Customer Data
# -----------------------------

st.subheader("Customer Data")

st.dataframe(df)