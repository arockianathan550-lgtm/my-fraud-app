import streamlit as st
import joblib
import pandas as pd
import json

# Load model & features
model = joblib.load('churn_model.pkl')
with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

# Page setup
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# Header
st.title("📊 Customer Churn Predictor")
st.markdown("### Predict if a customer will leave!")
st.divider()

# Input Form
st.subheader("👤 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age", min_value=18, 
        max_value=80, value=30
    )
    gender = st.selectbox(
        "Gender", ['Male', 'Female']
    )
    tenure = st.number_input(
        "Tenure (Months)", 
        min_value=1, max_value=72, value=12
    )
    monthly_charges = st.number_input(
        "Monthly Charges (₹)",
        min_value=500, max_value=10000, 
        value=2000, step=100
    )
    total_charges = st.number_input(
        "Total Charges (₹)",
        min_value=1000, max_value=500000,
        value=24000, step=1000
    )

with col2:
    department = st.selectbox(
        "Department",
        ['Management', 'Sales', 
         'Support', 'Technical']
    )
    satisfaction = st.slider(
        "Satisfaction (1=Unhappy, 5=Happy)",
        min_value=1, max_value=5, value=3
    )
    num_complaints = st.number_input(
        "Number of Complaints",
        min_value=0, max_value=10, value=0
    )
    internet = st.selectbox(
        "Internet Service",
        ['DSL', 'Fiber', 'No']
    )
    contract = st.selectbox(
        "Contract Type",
        ['Monthly', 'Yearly', '2Year']
    )

st.divider()

# Predict Button
if st.button("🔮 Predict Churn!", 
             use_container_width=True):

    # Convert to numbers
    gender_n   = 1 if gender == 'Male' else 0
    dept_n     = {'Management':0,'Sales':1,
                  'Support':2,'Technical':3}[department]
    internet_n = {'DSL':0,'Fiber':1,'No':2}[internet]
    contract_n = {'Monthly':0,'Yearly':2,
                  '2Year':1}[contract]

    # Create input
    input_data = pd.DataFrame([[
        age, gender_n, tenure,
        monthly_charges, total_charges,
        dept_n, satisfaction,
        num_complaints, internet_n, contract_n
    ]], columns=feature_columns)

    # Predict
    result      = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()
    st.subheader("🎯 Prediction Result")

    # Show result
    col1, col2 = st.columns(2)
    with col1:
        st.metric("✅ Stay Probability",
                  f"{probability[0]*100:.1f}%")
    with col2:
        st.metric("🔴 Churn Probability",
                  f"{probability[1]*100:.1f}%")

    st.divider()

    if result == 1:
        st.error("🔴 THIS CUSTOMER WILL CHURN!")
        st.markdown("### 👉 Immediate Actions:")
        if satisfaction <= 2:
            st.warning(
                "😟 Very unhappy! "
                "Call customer immediately!"
            )
        if num_complaints >= 5:
            st.warning(
                "📞 Too many complaints! "
                "Resolve all issues!"
            )
        if monthly_charges > 3000:
            st.warning(
                "💰 High charges! "
                "Offer special discount!"
            )
        if contract == 'Monthly':
            st.warning(
                "📅 Monthly contract! "
                "Offer yearly plan!"
            )
    else:
        st.success("✅ THIS CUSTOMER WILL STAY!")
        st.markdown("### 👉 Keep them happy:")
        st.info("🌟 Reward their loyalty!")
        st.info("📈 Offer upgrade options!")
        st.info("😊 Keep service quality high!")

    # Risk Meter
    st.divider()
    st.subheader("📊 Risk Level")
    churn_pct = probability[1] * 100

    if churn_pct < 30:
        st.success(f"🟢 LOW RISK — {churn_pct:.1f}%")
    elif churn_pct < 60:
        st.warning(f"🟡 MEDIUM RISK — {churn_pct:.1f}%")
    else:
        st.error(f"🔴 HIGH RISK — {churn_pct:.1f}%")

    st.progress(int(churn_pct))

st.divider()
st.markdown(
    "*Built with Python, "
    "RandomForest & Streamlit* 🚀"
)