import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")
st.title("Telecom Customer Churn Dashboard")

model = joblib.load("churn_model.joblib")
metrics = json.load(open("metrics.json"))
data = pd.read_csv("telecom_customers.csv")

tab1, tab2, tab3 = st.tabs(["Overview", "Predict", "Batch scoring"])

with tab1:
    c = st.columns(4)
    c[0].metric("Customers", f"{metrics['rows']:,}")
    c[1].metric("Churn rate", f"{metrics['churn_rate']:.1%}")
    c[2].metric("Accuracy", metrics["accuracy"])
    c[3].metric("ROC-AUC", metrics["roc_auc"])
    st.subheader("Churn rate by contract")
    st.bar_chart(data.groupby("contract").churn.mean())
    st.subheader("Churn rate by internet service")
    st.bar_chart(data.groupby("internet_service").churn.mean())

with tab2:
    col1, col2 = st.columns(2)
    gender = col1.selectbox("Gender", ["Male", "Female"])
    senior = col1.selectbox("Senior citizen", [0, 1])
    tenure = col1.slider("Tenure (months)", 1, 72, 12)
    contract = col1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = col2.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    pay = col2.selectbox("Payment method", ["Electronic check", "Mobile money", "Bank transfer", "Credit card"])
    monthly = col2.number_input("Monthly charges", 15.0, 120.0, 60.0)
    calls = col2.number_input("Support calls", 0, 20, 2)
    if st.button("Predict churn"):
        row = pd.DataFrame([{
            "gender": gender, "senior_citizen": senior, "tenure_months": tenure,
            "contract": contract, "internet_service": internet, "payment_method": pay,
            "monthly_charges": monthly, "support_calls": calls,
            "total_charges": monthly * tenure}])
        p = model.predict_proba(row)[0, 1]
        st.metric("Churn probability", f"{p:.1%}")
        st.error("High risk of churn") if p >= 0.5 else st.success("Low risk of churn")

with tab3:
    up = st.file_uploader("Upload CSV with the same columns as telecom_customers.csv", type="csv")
    if up:
        d = pd.read_csv(up)
        X = d.drop(columns=[c for c in ["customer_id", "churn"] if c in d.columns])
        d["churn_probability"] = model.predict_proba(X)[:, 1]
        st.dataframe(d.sort_values("churn_probability", ascending=False))
        st.download_button("Download results", d.to_csv(index=False), "scored.csv")
