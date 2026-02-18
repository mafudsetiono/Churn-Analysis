import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle as pkl

# CONFIG
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# LOAD MODEL & SCALER
@st.cache_resource
def load_model():
    model = joblib.load("model/churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# SIDEBAR
st.sidebar.title("📌 Mode Selection")

mode = st.sidebar.radio(
    "Choose Mode:",
    ["Single Prediction", "Batch Prediction", "Impact Simulator", "Dashboard"]
)

# MODE ROUTING
if mode == "Single Prediction":
    st.title("🔮 Single Customer Prediction")

    st.write("Input customer details to predict churn probability.")

    col1, col2 = st.columns(2)

    with col1:
        international_plan = st.selectbox("International Plan", ["No", "Yes"])
        voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
        customer_service_calls = st.number_input("Customer Service Calls", min_value=0, step=1)

    with col2:
        total_day_charge = st.number_input("Total Day Charge", min_value=0.0)
        total_eve_charge = st.number_input("Total Evening Charge", min_value=0.0)

    if st.button("Predict Churn"):
        input_df = pd.DataFrame({
            "International plan": [international_plan],
            "Voice mail plan": [voice_mail_plan],
            "Customer service calls": [customer_service_calls],
            "Total day charge": [total_day_charge],
            "Total eve charge": [total_eve_charge]
        })

        # Encoding
        input_df["International plan"] = input_df["International plan"].map({"Yes":1, "No":0})
        input_df["Voice mail plan"] = input_df["Voice mail plan"].map({"Yes":1, "No":0})

        # Scaling
        input_scaled = scaler.transform(input_df)

        # Predict probability
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("📊 Prediction Result")
        st.metric("Churn Probability", f"{prob*100:.2f}%")

        # Risk category
        if prob < 0.4:
            risk = "Low Risk"
            color = "🟢"
        elif prob < 0.75:
            risk = "Medium Risk"
            color = "🟡"
        else:
            risk = "High Risk"
            color = "🔴"

        st.write(f"Risk Category: {color} **{risk}**")
        st.subheader("🔍 Model Explanation")

        # Ambil coefficient
        coefficients = model.coef_[0]
        feature_names = [
            "International plan",
            "Voice mail plan",
            "Customer service calls",
            "Total day charge",
            "Total eve charge"
        ]

        # Hitung kontribusi tiap fitur
        contributions = input_scaled[0] * coefficients

        explain_df = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": contributions
        })

        # Urutkan berdasarkan pengaruh terbesar
        explain_df = explain_df.sort_values(by="Contribution", ascending=False)

        st.dataframe(explain_df)

        st.subheader("📌 Smart Recommendation")

        # Risk level classification
        if prob < 0.4:
            risk = "Low Risk"
            st.success("Customer is low risk. Maintain service quality and monitor periodically.")

        elif prob < 0.75:
            risk = "Medium Risk"
            st.warning("Customer is medium risk. Consider engagement campaign.")

        else:
            risk = "High Risk"
            st.error("Customer is high risk. Immediate retention strategy required.")
        
        top_positive = explain_df.iloc[0]
        top_negative = explain_df.iloc[-1]

        st.write("### 📌 Key Drivers:")

        st.write(
            f"🔺 Biggest risk driver: **{top_positive['Feature']}** "
            f"(Contribution: {top_positive['Contribution']:.3f})"
        )

        st.write(
            f"🔻 Strongest protective factor: **{top_negative['Feature']}** "
            f"(Contribution: {top_negative['Contribution']:.3f})"
        )

        st.bar_chart(explain_df.set_index("Feature"))
        
        # Feature-Based Insights
        recommendations = []

        # Customer Service Calls
        if customer_service_calls >= 3:
            recommendations.append(
                "High number of customer service calls detected → Immediate proactive support follow-up recommended."
            )

        # Day Charge
        if total_day_charge > 50:
            recommendations.append(
                "High day usage cost → Offer customized discount or bundle optimization."
            )

        # Evening Charge 
        if total_eve_charge > 30:
            recommendations.append(
                "High evening usage cost → Consider night package optimization."
            )

        # International Plan
        if international_plan == "Yes":
            recommendations.append(
                "International plan user → Review pricing competitiveness and international service quality."
            )

        # Voice Mail Plan
        if voice_mail_plan == "No":
            recommendations.append(
                "Customer not subscribed to voice mail plan → Upsell bundled feature to increase engagement."
            )

        # Display All Recommendations
        if recommendations:
            st.write("### 🎯 Personalized Action Plan:")
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.write("No critical behavioral risk indicators detected.")

        st.subheader("💰 Financial Impact Simulation")

        # Assumptions
        loss_per_customer = 50
        intervention_cost = 10
        success_rate = 0.8

        expected_loss = prob * loss_per_customer
        expected_saved = prob * success_rate * loss_per_customer
        net_benefit = expected_saved - intervention_cost

        col1, col2, col3 = st.columns(3)

        col1.metric("Expected Loss", f"$ {expected_loss:,.2f}")
        col2.metric("Expected Saved (80%)", f"$ {expected_saved:,.2f}")
        col3.metric("Net Benefit", f"$ {net_benefit:,.2f}")

        break_even_prob = intervention_cost / (success_rate * loss_per_customer)
        st.write(f"Break-even Probability: {break_even_prob*100:.2f}%")

        st.subheader("🚨 Retention Priority")

        if net_benefit > 20:
            st.error("🟥 PRIORITY 1 – Immediate Action Required")
            st.write("Customer shows high financial risk. Strongly recommended to initiate retention campaign immediately.")

        elif net_benefit > 0:
            st.warning("🟧 PRIORITY 2 – Strategic Intervention")
            st.write("Customer intervention is economically justified, but not critical.")

        else:
            st.success("🟩 PRIORITY 3 – Low Priority")
            st.write("Intervention is not economically efficient at this time.")



elif mode == "Batch Prediction":
    st.title("📂 Batch Prediction")

    st.write("Upload CSV file for bulk churn prediction.")

elif mode == "Impact Simulator":
    st.title("💰 Impact Simulator")

    st.write("Simulate financial impact of churn prevention strategy.")

elif mode == "Dashboard":
    st.title("📊 Churn Analytics Dashboard")

    st.write("Overview of churn insights and model performance.")
