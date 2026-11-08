import os
import joblib
import pandas as pd
import streamlit as st

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📞",
    layout="wide"
)

# Title & Description
st.title("📞 Telecom Customer Churn Prediction")
st.markdown("A simple, user-friendly Machine Learning app to predict whether a customer will stay or leave.")

# ==============================================================================
# Load Saved Model
# ==============================================================================
@st.cache_resource
def load_model():
    model_paths = [
        "models/customer_churn_model.joblib",
        os.path.join(os.path.dirname(__file__), "models", "customer_churn_model.joblib"),
        "customer_churn_model.joblib",
        os.path.join(os.path.dirname(__file__), "customer_churn_model.joblib")
    ]
    for p in model_paths:
        if os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError("Model file (customer_churn_model.joblib) not found!")

try:
    model = load_model()
    st.sidebar.success("Model loaded successfully! (Tuned XGBoost)")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# ==============================================================================
# Quick Sample Pre-fills (Sidebar)
# ==============================================================================
st.sidebar.header("⚡ Quick Test Profiles")
profile = st.sidebar.radio(
    "Pre-fill with a sample customer:",
    ["Default", "High-Risk Customer (Likely to Leave)", "Loyal Customer (Likely to Stay)"]
)

# Preset values based on selection
if profile == "High-Risk Customer (Likely to Leave)":
    default_gender = "Male"
    default_senior = 0
    default_partner = "No"
    default_dependents = "No"
    default_tenure = 2
    default_phone = "Yes"
    default_multilines = "No"
    default_internet = "Fiber optic"
    default_security = "No"
    default_backup = "No"
    default_device = "No"
    default_support = "No"
    default_tv = "No"
    default_movies = "No"
    default_contract = "Month-to-month"
    default_paperless = "Yes"
    default_payment = "Electronic check"
    default_monthly = 85.0
elif profile == "Loyal Customer (Likely to Stay)":
    default_gender = "Female"
    default_senior = 0
    default_partner = "Yes"
    default_dependents = "Yes"
    default_tenure = 60
    default_phone = "Yes"
    default_multilines = "Yes"
    default_internet = "DSL"
    default_security = "Yes"
    default_backup = "Yes"
    default_device = "Yes"
    default_support = "Yes"
    default_tv = "Yes"
    default_movies = "Yes"
    default_contract = "Two year"
    default_paperless = "No"
    default_payment = "Credit card (automatic)"
    default_monthly = 65.0
else:
    default_gender = "Female"
    default_senior = 0
    default_partner = "Yes"
    default_dependents = "No"
    default_tenure = 12
    default_phone = "Yes"
    default_multilines = "No"
    default_internet = "DSL"
    default_security = "No"
    default_backup = "Yes"
    default_device = "No"
    default_support = "No"
    default_tv = "No"
    default_movies = "No"
    default_contract = "Month-to-month"
    default_paperless = "Yes"
    default_payment = "Electronic check"
    default_monthly = 55.0

# ==============================================================================
# Customer Input Form
# ==============================================================================
st.subheader("📋 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Demographics & Account")
    gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(default_gender))
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=default_senior)
    partner = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(default_partner))
    dependents = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(default_dependents))
    tenure = st.slider("Tenure (Months with company)", min_value=1, max_value=72, value=default_tenure)

with col2:
    st.markdown("#### 🌐 Subscribed Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(default_phone))
    multi_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(default_multilines))
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(default_internet))
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_support))
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_security))
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_backup))
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_device))
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_tv))
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_movies))

with col3:
    st.markdown("#### 💳 Billing & Contract")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(default_contract))
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(default_paperless))
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(default_payment)
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=130.0, value=float(default_monthly), step=0.5)
    
    # Auto-calculate estimated TotalCharges based on tenure and monthly
    est_total = float(tenure * monthly_charges)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=est_total, step=5.0)

# ==============================================================================
# Helper: Determine Tenure Group
# ==============================================================================
def get_tenure_group(t):
    if t <= 12:
        return '0-1yr'
    elif t <= 24:
        return '1-2yr'
    elif t <= 48:
        return '2-4yr'
    elif t <= 60:
        return '4-5yr'
    else:
        return '5+yr'

# ==============================================================================
# Predict Action
# ==============================================================================
st.markdown("---")

if st.button("🔮 Predict Customer Churn Risk", type="primary", use_container_width=True):
    # Construct customer input DataFrame
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multi_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': get_tenure_group(tenure)
    }])
    
    # Run prediction
    pred_class = model.predict(input_data)[0]
    churn_prob = model.predict_proba(input_data)[0, 1]
    
    st.markdown("### 📊 Prediction Results")
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if churn_prob >= 0.40:
            st.error(f"🚨 **HIGH CHURN RISK**")
            st.markdown(f"**This customer is likely to cancel their service.**")
        else:
            st.success(f"✅ **LOW CHURN RISK (LOYAL)**")
            st.markdown(f"**This customer is likely to remain with the company.**")
            
        st.metric(label="Predicted Churn Probability", value=f"{churn_prob * 100:.1f}%")
        st.progress(float(churn_prob))
        
    with res_col2:
        st.markdown("#### 💡 Retention Strategy Recommendations")
        if churn_prob >= 0.40:
            st.warning("Recommended actions to retain this customer:")
            if contract == "Month-to-month":
                st.markdown("- 🏷️ **Offer Contract Discount**: Offer 10-15% discount for a 1-year contract.")
            if tech_support == "No" and internet_service != "No":
                st.markdown("- 🛠️ **Complimentary Tech Support**: Provide 3 months of free premium tech support.")
            if payment == "Electronic check":
                st.markdown("- 💳 **Auto-Pay Incentive**: Offer $5 credit to switch to automatic card/bank payment.")
            if monthly_charges > 75:
                st.markdown("- 📦 **Plan Optimization**: Suggest bundle options to lower monthly bills.")
        else:
            st.info("Customer is satisfied and loyal:")
            st.markdown("- ⭐ **Loyalty Perks**: Send loyalty appreciation emails or referral bonuses.")
            st.markdown("- 🎁 **Upsell Potential**: Eligible for premium speed upgrades or add-on entertainment.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Machine Learning Model: Tuned XGBoost with Preprocessing Pipeline")
