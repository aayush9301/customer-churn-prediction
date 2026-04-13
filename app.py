"""
Customer Churn Prediction — Streamlit Web App
============================================
Run this after training the model in the notebook.
Command: streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import os

# ─── Page Config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .churn-box {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .safe-box {
        background: linear-gradient(135deg, #27ae60, #1e8449);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin-bottom: 0.8rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.3rem;
        margin: 1.2rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model  = joblib.load('models/churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/feature_names.json') as f:
        features = json.load(f)
    return model, scaler, features


# ─── Header ─────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🔮 Customer Churn Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict whether a customer will leave using Machine Learning</p>', unsafe_allow_html=True)

# ─── Model loading check ─────────────────────────────────────────────────────

if not os.path.exists('models/churn_model.pkl'):
    st.error("❌ Model not found! Please run the Jupyter notebook first to train and save the model.")
    st.info("📌 Steps:\n1. Open `customer_churn_prediction.ipynb`\n2. Run all cells\n3. Come back here")
    st.stop()

model, scaler, feature_names = load_model()
st.sidebar.success("✅ Model loaded successfully!")


# ─── Sidebar — Customer Input ────────────────────────────────────────────────

st.sidebar.markdown("## 📋 Customer Details")
st.sidebar.markdown("Fill in the customer's information below:")

# Demographics
st.sidebar.markdown('<p class="section-header">👤 Demographics</p>', unsafe_allow_html=True)
gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior         = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner        = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents     = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

# Account Info
st.sidebar.markdown('<p class="section-header">💼 Account Info</p>', unsafe_allow_html=True)
tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract       = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment        = st.sidebar.selectbox("Payment Method", [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                 ])

# Services
st.sidebar.markdown('<p class="section-header">📡 Services</p>', unsafe_allow_html=True)
phone_service  = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multi_lines    = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet       = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
online_sec     = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup  = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_prot    = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support   = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
stream_tv      = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
stream_movies  = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# Billing
st.sidebar.markdown('<p class="section-header">💰 Billing</p>', unsafe_allow_html=True)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
total_charges   = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0,
                                           float(tenure * monthly_charges), step=10.0)


# ─── Encode Input ────────────────────────────────────────────────────────────

def encode_input():
    # Binary mappings
    yn = {'Yes': 1, 'No': 0}
    three_val = {'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0}

    # Scale numeric
    numeric = scaler.transform([[tenure, monthly_charges, total_charges]])

    raw = {
        'gender'                       : 1 if gender == 'Male' else 0,
        'SeniorCitizen'                : yn[senior],
        'Partner'                      : yn[partner],
        'Dependents'                   : yn[dependents],
        'tenure'                       : numeric[0][0],
        'PhoneService'                 : yn[phone_service],
        'MultipleLines'                : three_val[multi_lines],
        'OnlineSecurity'               : three_val[online_sec],
        'OnlineBackup'                 : three_val[online_backup],
        'DeviceProtection'             : three_val[device_prot],
        'TechSupport'                  : three_val[tech_support],
        'StreamingTV'                  : three_val[stream_tv],
        'StreamingMovies'              : three_val[stream_movies],
        'PaperlessBilling'             : yn[paperless],
        'MonthlyCharges'               : numeric[0][1],
        'TotalCharges'                 : numeric[0][2],
        # Internet Service (one-hot, drop_first = DSL is baseline)
        'InternetService_Fiber optic'  : 1 if internet == 'Fiber optic' else 0,
        'InternetService_No'           : 1 if internet == 'No' else 0,
        # Contract (drop_first = Month-to-month is baseline)
        'Contract_One year'            : 1 if contract == 'One year' else 0,
        'Contract_Two year'            : 1 if contract == 'Two year' else 0,
        # Payment Method (drop_first = Bank transfer is baseline)
        'PaymentMethod_Credit card (automatic)' : 1 if payment == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check'        : 1 if payment == 'Electronic check' else 0,
        'PaymentMethod_Mailed check'            : 1 if payment == 'Mailed check' else 0,
    }

    # Align with trained feature order
    input_df = pd.DataFrame([raw])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]
    return input_df


# ─── Main Panel ─────────────────────────────────────────────────────────────

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📊 Customer Overview")

    # Display entered details as a nice table
    details = {
        "Tenure"         : f"{tenure} months",
        "Contract"       : contract,
        "Internet"       : internet,
        "Monthly Charges": f"${monthly_charges:.2f}",
        "Total Charges"  : f"${total_charges:.2f}",
        "Payment"        : payment,
        "Tech Support"   : tech_support,
        "Online Security": online_sec,
    }
    detail_df = pd.DataFrame(details.items(), columns=["Feature", "Value"])
    st.dataframe(detail_df, use_container_width=True, hide_index=True)


with col2:
    st.markdown("### 🔮 Prediction")
    
    predict_btn = st.button("🚀 Predict Churn", use_container_width=True, type="primary")

    if predict_btn:
        input_data  = encode_input()
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        churn_prob  = probability[1] * 100
        stay_prob   = probability[0] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="churn-box">
                ⚠️ HIGH CHURN RISK<br>
                <span style="font-size:2rem">{churn_prob:.1f}%</span><br>
                <small>probability of churning</small>
            </div>""", unsafe_allow_html=True)
            st.warning("💡 **Recommended actions:** Offer a discount, upgrade to yearly plan, or reach out proactively.")
        else:
            st.markdown(f"""
            <div class="safe-box">
                ✅ LIKELY TO STAY<br>
                <span style="font-size:2rem">{stay_prob:.1f}%</span><br>
                <small>probability of staying</small>
            </div>""", unsafe_allow_html=True)
            st.info("💡 **Customer seems loyal.** Continue providing good service.")

        # Probability bar
        st.markdown("#### Probability Breakdown")
        st.progress(int(stay_prob), text=f"Stay: {stay_prob:.1f}%")
        st.progress(int(churn_prob), text=f"Churn: {churn_prob:.1f}%")

        # Risk meter
        st.markdown("#### Risk Level")
        if churn_prob < 30:
            st.success(f"🟢 LOW RISK ({churn_prob:.1f}%)")
        elif churn_prob < 60:
            st.warning(f"🟡 MEDIUM RISK ({churn_prob:.1f}%)")
        else:
            st.error(f"🔴 HIGH RISK ({churn_prob:.1f}%)")
    else:
        st.info("👈 Fill in the customer details in the sidebar, then click **Predict Churn**.")


# ─── Bottom Section — Risk Factors ──────────────────────────────────────────

st.markdown("---")
st.markdown("### 📌 Common Churn Risk Factors")

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.markdown("""
    **📅 Short Tenure**  
    Customers in their first 1-12 months are most likely to churn.
    """)

with r2:
    st.markdown("""
    **📄 Month-to-Month**  
    No long-term commitment = easier to leave anytime.
    """)

with r3:
    st.markdown("""
    **💸 High Monthly Bill**  
    Customers paying $70+ monthly are more price-sensitive.
    """)

with r4:
    st.markdown("""
    **🔒 No Security/Support**  
    Missing tech support or online security increases dissatisfaction.
    """)


# ─── Footer ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Customer Churn Prediction | ML Project | Built with Scikit-learn + XGBoost + Streamlit"
    "</p>",
    unsafe_allow_html=True
)
