import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    model = pickle.load(open(BASE / "model.pkl", "rb"))
    meta = pickle.load(open(BASE / "metadata.pkl", "rb"))
    return model, meta


model, meta = load_artifacts()
opts = meta["cat_options"]

st.title("📉 Telco Customer Churn Predictor")
st.write(
    "Enter a customer's account details to estimate the probability that they will **cancel their service** (churn). "
    "The model is a tuned Random Forest trained on the IBM Telco Customer Churn dataset."
)

st.subheader("Customer details")
col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox("Contract", opts["Contract"], index=opts["Contract"].index("Month-to-month"))
    tenure = st.slider("Tenure (months with the company)", 0, 72, 12)
    monthly = st.number_input("Monthly charges ($)", min_value=15.0, max_value=125.0, value=70.0, step=0.5)

with col2:
    internet = st.selectbox("Internet service", opts["InternetService"],
                            index=opts["InternetService"].index("Fiber optic"))
    payment = st.selectbox("Payment method", opts["PaymentMethod"],
                           index=opts["PaymentMethod"].index("Electronic check"))

if st.button("Predict churn", type="primary"):
    row = pd.DataFrame([{
        "Contract": contract,
        "tenure": tenure,
        "InternetService": internet,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
    }])[meta["features"]]  # same columns/order as in training

    proba = float(model.predict_proba(row)[0, 1])
    pred = int(model.predict(row)[0])

    st.subheader("Result")
    st.metric("Churn probability", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))
    if pred == 1:
        st.error("⚠️ **Likely to churn.** Consider a retention offer (e.g., a longer contract or a discount).")
    else:
        st.success("✅ **Likely to stay.**")

with st.expander("About the model"):
    st.write(f"**Best hyperparameters:** {meta['best_params']}")
    st.write("**Test-set performance:**")
    st.json(meta["metrics"])
    st.caption("This is a demonstration model trained on a sample dataset, not production advice.")
