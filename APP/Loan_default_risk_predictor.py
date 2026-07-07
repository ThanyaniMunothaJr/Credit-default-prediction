"""
Loan Default Prediction App
----------------------------
A simple front-end for tellers/staff to enter a customer's loan details
and get a default-risk prediction from the trained model.

Run with:
    streamlit run Loan_default_risk_predictor.py

Requires the model file saved earlier in the notebook, e.g.:
    default_pred_log_with_threshold.pkl
(a dict containing {"model": model_name, "threshold": float, "target_recall": float})

Place this .py file in the SAME folder as that .pkl file, or update MODEL_PATH below.
"""

import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "XGB_default_pred_model.pkl"

st.set_page_config(page_title="Loan Default Predictor", page_icon="\U0001F4B0", layout="centered")

# ---------------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    bundle = joblib.load(path)
    # Support both a plain pipeline OR a dict bundle with a threshold
    if isinstance(bundle, dict):
        model = bundle.get("model")
        threshold = bundle.get("threshold", 0.5)
    else:
        model = bundle
        threshold = 0.5
    return model, threshold


loaded = load_model(MODEL_PATH)

st.title("Loan Default Risk Predictor")
st.caption("Enter the applicant's loan and credit details below to estimate default risk.")

if loaded is None:
    st.error(
        f"Could not find model file '{MODEL_PATH}'. "
        f"Make sure it's in the same folder as this app, or update MODEL_PATH in the script."
    )
    st.stop()

model, default_threshold = loaded

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Loan Terms")

col_a, col_b, col_c = st.columns(3)
with col_a:
    loan_amnt = st.number_input("Loan Amount (R)", min_value=500, max_value=None, value=500, step=500)
with col_b:
    term = st.selectbox("Term (months)", options=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84], index=0)
with col_c:
    int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=12.5, step=0.1)


def calculate_installment(principal, annual_rate_pct, term_months):
    """Standard amortized loan payment formula (fixed monthly payment)."""
    monthly_rate = (annual_rate_pct / 100) / 12
    if monthly_rate == 0:
        return principal / term_months
    return principal * (monthly_rate * (1 + monthly_rate) ** term_months) / (
        (1 + monthly_rate) ** term_months - 1
    )


installment = calculate_installment(loan_amnt, int_rate, term)

st.info(f"Calculated Monthly Installment: R{installment:,.2f}")

st.markdown("---")

with st.form("loan_form"):
    st.subheader("Applicant / Credit Details")

    col1, col2 = st.columns(2)

    with col1:
        annual_inc = st.number_input("Annual Income (R)", min_value=0.0, max_value=1_000_000.0, value=60000.0, step=1000.0)

    with col2:
        dti = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=100.0, value=18.0, step=0.5)
        grade = st.selectbox("Loan Grade", options=["A", "B", "C", "D", "E", "F", "G"], index=2)
        open_acc = st.number_input("Number of Open Credit Accounts", min_value=0, max_value=100, value=10, step=1)
        revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=200.0, value=45.0, step=1.0)
        credit_history_years = st.number_input("Credit History Length (years)", min_value=0.0, max_value=80.0, value=10.0, step=0.5)

    st.markdown("---")
    threshold = st.slider(
        "Decision threshold (probability cutoff for 'high risk')",
        min_value=0.05, max_value=0.95, value=float(default_threshold), step=0.01,
        help="Lower threshold = catches more defaulters but flags more false alarms. "
             "Default value below is the one tuned during model development."
    )

    submitted = st.form_submit_button("Predict Default Risk")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "annual_inc": annual_inc,
        "dti": dti,
        "grade": grade,
        "open_acc": open_acc,
        "revol_util": revol_util,
        "credit_history_years": credit_history_years,
    }])

    proba = model.predict_proba(input_df)[:, 1][0]
    prediction = int(proba >= threshold)

    st.markdown("---")
    st.subheader("Result")

    col1, col2 = st.columns(2)
    col1.metric("Predicted Default Probability", f"{proba:.1%}")
    col2.metric("Decision Threshold", f"{threshold:.2f}")

    if prediction == 1:
        st.error("⚠️ HIGH RISK — model predicts this applicant is likely to default.")
    else:
        st.success("✅ LOW RISK — model predicts this applicant is likely to repay in full.")

    with st.expander("What does this mean?"):
        st.write(
            "This model estimates the probability that a loan will end in default "
            "('Charged Off') based on historical LendingClub loan outcomes. "
            "A prediction is not a guarantee — it reflects patterns in past data, "
            "and should be used as one input alongside other underwriting judgment, "
            "not as the sole basis for a lending decision."
        )
        st.write(
            "Lowering the threshold makes the model flag more applicants as high risk "
            "(catching more true defaulters, but also more false alarms on safe borrowers). "
            "Raising it does the opposite. The default value shown was tuned during model "
            "development to target a specific recall level on historical test data — "
            "adjust the slider above to see how the prediction changes at other cutoffs."
        )

st.markdown("---")
st.caption(
    "Model: Logistic Regression trained on LendingClub historical loan data. "
    "For internal decision-support use only — not a substitute for full underwriting review."
)
