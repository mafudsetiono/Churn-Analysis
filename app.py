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
    ["Profile", "Dashboard", "Single Prediction", "Batch Prediction"]
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

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("Preview Data:")
        st.dataframe(df.head())
        required_features = [
            "International plan",
            "Voice mail plan",
            "Customer service calls",
            "Total day charge",
            "Total eve charge"
        ]

        df_model = df[required_features].copy()

        # Encoding
        df_model["International plan"] = df_model["International plan"].map({"Yes":1, "No":0})
        df_model["Voice mail plan"] = df_model["Voice mail plan"].map({"Yes":1, "No":0})

        # Scaling
        df_scaled = scaler.transform(df_model)

        # Predict probability
        probabilities = model.predict_proba(df_scaled)[:,1]

        df["Churn Probability"] = probabilities

        loss_per_customer = 50
        intervention_cost = 10
        success_rate = 0.8

        df["Expected Loss"] = df["Churn Probability"] * loss_per_customer
        df["Expected Saved"] = df["Churn Probability"] * success_rate * loss_per_customer
        df["Net Benefit"] = df["Expected Saved"] - intervention_cost

        # Retention Priority
        conditions = [
            df["Net Benefit"] > 20,
            df["Net Benefit"] > 0
        ]

        choices = [
            "Priority 1 - Immediate",
            "Priority 2 - Strategic"
        ]

        df["Retention Priority"] = np.select(
            conditions,
            choices,
            default="Priority 3 - Low"
        )
        # Sort by Net Benefit   
        df_sorted = df.sort_values(by="Net Benefit", ascending=False)

        st.subheader("📊 Retention Ranking")
        st.dataframe(df_sorted.head(10))


elif mode == "Dashboard":

    st.title("📊 Churn Risk Dashboard")

    # Load Dataset
    df = pd.read_csv("data/churn-bigml-20.csv")

    required_features = [
        "International plan",
        "Voice mail plan",
        "Customer service calls",
        "Total day charge",
        "Total eve charge"
    ]

    df_model = df[required_features].copy()

    # Encoding
    df_model["International plan"] = df_model["International plan"].map({"Yes":1, "No":0})
    df_model["Voice mail plan"] = df_model["Voice mail plan"].map({"Yes":1, "No":0})

    # Scaling
    df_scaled = scaler.transform(df_model)

    # Predict Probability
    probabilities = model.predict_proba(df_scaled)[:,1]
    df["Churn Probability"] = probabilities

    # Financial Assumptions
    loss_per_customer = 50
    intervention_cost = 10
    success_rate = 0.8

    df["Expected Loss"] = df["Churn Probability"] * loss_per_customer
    df["Expected Saved"] = df["Churn Probability"] * success_rate * loss_per_customer
    df["Net Benefit"] = df["Expected Saved"] - intervention_cost

    # Retention Priority
    conditions = [
        df["Net Benefit"] > 20,
        df["Net Benefit"] > 0.5
    ]

    choices = [
        "Priority 1",
        "Priority 2"
    ]

    df["Retention Priority"] = np.select(
        conditions,
        choices,
        default="Priority 3"
    )

    # Executive KPIs
    st.subheader("📌 Executive Overview")

    total_customers = len(df)
    avg_prob = df["Churn Probability"].mean()
    total_expected_loss = df["Expected Loss"].sum()
    total_potential_saving = df[df["Net Benefit"] > 0]["Net Benefit"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)
    col2.metric("Avg Churn Probability", f"{avg_prob*100:.2f}%")
    col3.metric("Total Expected Loss", f"$ {total_expected_loss:,.2f}")
    col4.metric("Total Potential Saving", f"$ {total_potential_saving:,.2f}")

    st.divider()

    priority1_customers = df[df["Retention Priority"] == "Priority 1"]

    total_intervention_cost = len(priority1_customers) * intervention_cost
    total_priority1_net = priority1_customers["Net Benefit"].sum()

    if total_intervention_cost > 0:
        roi = total_priority1_net / total_intervention_cost
    else:
        roi = 0

    st.subheader("💰 Retention ROI Analysis")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers in Priority 1", len(priority1_customers))
    col2.metric("Total Intervention Cost", f"$ {total_intervention_cost:,.2f}")
    col3.metric("Estimated ROI", f"{roi:.2f}x")
    col4.metric("Net Benefit from Priority 1", f"$ {total_priority1_net:,.2f}")


    # Risk Distribution
    st.subheader("📊 Risk Distribution")

    risk_bins = pd.cut(
        df["Churn Probability"],
        bins=[0, 0.4, 0.75, 1],
        labels=["Low", "Medium", "High"]
    )

    risk_counts = risk_bins.value_counts().sort_index()

    st.bar_chart(risk_counts)

    st.divider()

    # Top Retention Targets
    st.subheader("🚨 Top Retention Targets")

    df_top = df.sort_values(by="Net Benefit", ascending=False).head(10)

    st.dataframe(
        df_top[[
            "Churn Probability",
            "Expected Loss",
            "Net Benefit",
            "Retention Priority"
        ]]
    )

    st.divider()

    # Executive Summary
    st.subheader("📌 Executive Summary")

    high_risk_count = (df["Retention Priority"] == "Priority 1").sum()
    medium_risk_count = (df["Retention Priority"] == "Priority 2").sum()

    st.write(f"""
    - The portfolio consists of **{total_customers} customers**.
    - The average churn probability is **{avg_prob*100:.2f}%**, indicating moderate overall churn risk.
    - Estimated total potential revenue loss is **${total_expected_loss:,.2f}**.
    - Targeted intervention could generate up to **${total_potential_saving:,.2f} in net benefit**.
    - **{high_risk_count} customers** require immediate retention action.
    - **{medium_risk_count} customers** qualify for strategic intervention.
    """)

    st.divider()

    st.subheader("🎯 Recommended Action Plan")

    st.markdown("""
    ### 🔴 Priority 1 – Immediate Action
    - Launch personalized retention offers.
    - Assign senior customer service representatives.
    - Offer discount or loyalty incentives.

    ### 🟠 Priority 2 – Strategic Engagement
    - Run targeted engagement campaigns.
    - Monitor behavior closely for next 30 days.
    - Provide value-based communication.

    ### 🟢 Priority 3 – Maintain & Monitor
    - Maintain service quality.
    - No immediate financial intervention required.
    """)
    
elif mode == "Profile":
    
    # Profile Section
    st.title("👤 Profile")

    st.caption("Data Analyst | Data Scientist | Business Insight Enthusiast | Data Engineering")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(
            "assets/Foto Nonformal.png",
            width=220,
            caption="Mafud Satrio Setiono"
        )

    with col2:
        st.subheader("Halo, saya Mafud 👋")

        st.markdown("""
            Saya **Mafud Satrio Setiono**, seorang **Data Enthusiast** yang berfokus pada
            **Data Engineering, Data Analysis, Data Science, dan Business Insight**.

            Saya memiliki latar belakang **Teknik Informatika** dan pengalaman mengerjakan
            berbagai proyek analisis data menggunakan **Python, SQL**, serta tools visualisasi
            seperti **Tableau dan Power BI**.

            Saya tertarik membangun solusi berbasis data yang **tidak hanya akurat secara teknis,
            tetapi juga relevan untuk pengambilan keputusan bisnis**.
        """)

    # Skills Section
    st.divider()
    st.subheader("🛠️ Skills & Tools")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            **📊 Data & BI Tools**
            - Tableau  
            - Power BI  
            - Looker
            - Excel  
            - Git & GitHub
        """)

    with col2:
        st.markdown("""
            **💻 Programming & Database**
            - Python  
            - PostgreSQL  
            - MySQL
        """)

    with col3:
        st.markdown("""
            **🤝 Soft Skills**
            - Problem Solving  
            - Critical Thinking  
            - Communication  
            - Teamwork  
            - Time Management  
            - Adaptability  
        """)

    # Value Proposition
    st.divider()
    st.subheader("💡 What I Bring")

    st.markdown("""
        - Mampu menerjemahkan data menjadi **insight yang actionable**  
        - Fokus pada **business impact**, bukan hanya model  
        - Terbiasa bekerja dengan **data historis & time series**  
        - Berpikir strategis dan berbasis ROI  
        - Berkomunikasi efektif dengan pemangku kepentingan non-teknis
    """)

    # Contact Section
    st.divider()
    st.subheader("📬 Contact & Links")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            📧 **Email**  
            riosetiono23@gmail.com
        """)

    with col2:
        st.markdown("""
            🔗 **LinkedIn**  
            [linkedin.com/in/mafud-satrio-setiono](https://www.linkedin.com/in/mafud-satrio-setiono-5950a7266/)
    """)

    st.caption("Terbuka untuk peluang Data Analyst / Data Scientist ")

