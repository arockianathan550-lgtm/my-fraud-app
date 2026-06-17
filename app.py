import streamlit as st
import joblib
import pandas as pd
import json
import numpy as np

# Load model & features
model = joblib.load('fraud_model.pkl')
with open('fraud_features.json', 'r') as f:
    feature_cols = json.load(f)

# Page setup
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🏦",
    layout="centered"
)

# Header
st.title("🏦 Bank Fraud Detection System")
st.markdown("### Detect suspicious transactions instantly!")
st.divider()

# Input Form
st.subheader("💳 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Customer Age",
        min_value=18, max_value=80, value=30
    )
    balance = st.number_input(
        "Account Balance (₹)",
        min_value=1000, max_value=1000000,
        value=50000, step=1000
    )
    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=100, max_value=200000,
        value=5000, step=100
    )
    hour = st.slider(
        "Transaction Hour (0=Midnight, 23=11PM)",
        min_value=0, max_value=23, value=14
    )

with col2:
    num_transactions = st.number_input(
        "Transactions Today",
        min_value=1, max_value=20, value=2
    )
    distance = st.number_input(
        "Distance from Home (km)",
        min_value=0, max_value=1000, value=10
    )
    trans_type = st.selectbox(
        "Transaction Type",
        ['ATM', 'Online', 'POS', 'Transfer']
    )
    device = st.selectbox(
        "Device Used",
        ['ATM_Machine', 'Desktop', 'Mobile']
    )

st.divider()

# Warning indicators
st.subheader("⚠️ Risk Indicators")
col1, col2, col3, col4 = st.columns(4)

is_night = hour <= 4
high_amt = amount > 70000
far_home = distance > 700
many_trans = num_transactions > 15

with col1:
    if is_night:
        st.error("🌙 Night Time!")
    else:
        st.success("☀️ Day Time")

with col2:
    if high_amt:
        st.error("💰 High Amount!")
    else:
        st.success("💵 Normal Amount")

with col3:
    if far_home:
        st.error("📍 Far from Home!")
    else:
        st.success("🏠 Near Home")

with col4:
    if many_trans:
        st.error("🔄 Too Many Today!")
    else:
        st.success("✅ Normal Count")

st.divider()

# Predict Button
if st.button("🔍 Analyze Transaction!",
             use_container_width=True):

    # Convert to numbers
    type_map = {'ATM':0,'Online':1,
                'POS':2,'Transfer':3}
    device_map = {'ATM_Machine':0,
                  'Desktop':1,'Mobile':2}

    suspicious = 1 if (is_night and
                       high_amt and
                       far_home) else 0

    input_data = pd.DataFrame([[
        age, balance, amount, hour,
        num_transactions, distance,
        type_map[trans_type],
        device_map[device],
        int(is_night), int(high_amt),
        int(far_home), int(suspicious)
    ]], columns=feature_cols)

    result = model.predict(input_data)[0]
    prob   = model.predict_proba(input_data)[0]

    fraud_pct  = prob[1] * 100
    normal_pct = prob[0] * 100

    st.divider()
    st.subheader("🎯 Analysis Result")

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("✅ Normal Probability",
                  f"{normal_pct:.1f}%")
    with col2:
        st.metric("🔴 Fraud Probability",
                  f"{fraud_pct:.1f}%")

    # Risk meter
    st.progress(int(fraud_pct))

    st.divider()

    if result == 1:
        st.error("🚨 FRAUD ALERT! BLOCK THIS TRANSACTION!")
        st.markdown("### 👉 Immediate Actions:")
        if is_night:
            st.warning(
                "🌙 Midnight transaction detected!"
            )
        if high_amt:
            st.warning(
                "💰 Unusually high amount!"
            )
        if far_home:
            st.warning(
                "📍 Transaction far from home location!"
            )
        if many_trans:
            st.warning(
                "🔄 Too many transactions today!"
            )
        st.error(
            "📞 Contact customer immediately!"
        )
    else:
        st.success("✅ TRANSACTION APPROVED!")
        if fraud_pct > 30:
            st.warning(
                "⚠️ Medium risk — Monitor this account"
            )
        else:
            st.info(
                "😊 Normal transaction pattern!"
            )

st.divider()
st.markdown(
    "*Built with Python, "
    "Random Forest & Streamlit* 🏦"
)