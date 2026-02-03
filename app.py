import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.title("🏢 EDI Control Tower - Predictive Analytics Dashboard")
st.set_page_config(
    page_title="🏢 EDI Control Tower - Predictive Analytics Dashboard",
    layout="wide"
)
st.caption("Predictive Risk • Data Quality • Processing Time Intelligence")

# ---------------------------------------------------
# Styles
# ---------------------------------------------------
st.markdown("""
<style>
.kpi {
    border-radius: 14px;
    padding: 25px;
    text-align: center;
    color: white;
    height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
.kpi h2 { font-size: 36px; margin: 0; }
.kpi p { font-size: 15px; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Header
# ---------------------------------------------------
#st.title("🚀 EDI Predictive Analytics Dashboard")
st.markdown("""
**Scope:**  
Data Quality Scoring • Failure Risk Prediction • Processing Time Estimation  
*(Synthetic data – academic demonstration)*
""")

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
st.sidebar.markdown("## 📌 Navigation")

page = st.sidebar.radio(
    "Select View",
    [
        "📊 Operational Dashboard",
        "📈 DQ Score Distribution",
        "🚨 Failure Risk Levels",
        "⏱️ Processing Time & SLA Trends",
        "🧪 Data Lab (CSV / Synthetic)",
        "ℹ️ About Project"
    ]
)


# ---------------------------------------------------
# Synthetic Dataset
# ---------------------------------------------------
np.random.seed(42)
rows = 600

data = pd.DataFrame({
    "po_id": range(5001, 5001 + rows),
    "dq_score": np.random.randint(30, 100, rows),
    "missing": np.random.randint(0, 3, rows),
    "invalid_ref": np.random.randint(0, 2, rows),
    "format_err": np.random.randint(0, 2, rows),
    "partner_err": np.random.randint(0, 2, rows),
    "order_lines": np.random.randint(1, 20, rows)
})

data["order_failed"] = ((data["missing"] > 1) | (data["invalid_ref"] > 0)).astype(int)

data["processing_time_min"] = (
    5 + (100 - data["dq_score"]) * 0.3 +
    data["order_lines"] * 0.8 +
    data["order_failed"] * 10
).round(2)

# ---------------------------------------------------
# Model Training (cached)
# ---------------------------------------------------
@st.cache_resource
def train_models(data):
    
    features = [
        "dq_score",
        "missing",
        "invalid_ref",
        "format_err",
        "partner_err",
        "order_lines"
    ]

    X = data[features]
    y_fail = data["order_failed"]
    y_time = data["processing_time_min"]

    X_train, X_test, y_fail_train, y_fail_test = train_test_split(
        X, y_fail, test_size=0.3, random_state=42
    )

    _, _, y_time_train, y_time_test = train_test_split(
        X, y_time, test_size=0.3, random_state=42
    )

    # -------- Classification Models --------
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_fail_train)

    xgb = XGBClassifier(
        n_estimators=120,
        learning_rate=0.1,
        max_depth=4,
        eval_metric="logloss",
        verbosity=0,
        random_state=42
    )
    xgb.fit(X_train, y_fail_train)

    # -------- Regression Model (FIXED) --------
    time_model = XGBRegressor(
        n_estimators=120,
        learning_rate=0.1,
        max_depth=4,
        verbosity=0,
        random_state=42
    )
    time_model.fit(X_train, y_time_train)  # ✅ FIX

    rf_acc = accuracy_score(y_fail_test, rf.predict(X_test))
    xgb_acc = accuracy_score(y_fail_test, xgb.predict(X_test))

    return rf, xgb, time_model, rf_acc, xgb_acc

# ===================================================
# PAGE 1 – OPERATIONAL DASHBOARD
# ===================================================
if page == "📊 Operational Dashboard":

    st.subheader("📊 Operational Dashboard")

    total_pos = len(data)
    failed_pos = data.order_failed.sum()
    success_pos =total_pos - failed_pos
    processing_time = round(data.processing_time_min.mean(),2)
    
    c1, c2, c3, c4 = st.columns(4)
    #c1.metric("Total POs", total_pos)
    #c2.metric("Successful Orders", success_pos)
    #c3.metric("Failed Orders", failed_pos)
    #c4.metric("Avg Processing Time (min)", round(data.processing_time_min.mean(),2))

    with c1:
        st.markdown(f"""
        <div class="kpi" style="background:#4287f5;">
            <h2>{total_pos}</h2>
            <p>📦 Total POs</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi" style="background:#2ca02c;">
            <h2>{success_pos}</h2>
            <p>✅ Successful Orders</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="kpi" style="background:#b32400;">
            <h2>{failed_pos}</h2>
            <p>🚫 Failed Orders</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c4:
        st.markdown(f"""
        <div class="kpi" style="background:#e8cd1c;">
            <h2>{processing_time}</h2>
            <p>🕒 Avg Processing Time (min)</p>
        </div>
        """, unsafe_allow_html=True)

# ===================================================
# Order Success vs Failure – Donut Chart + Metric
# ===================================================

    import matplotlib.pyplot as plt

    st.subheader("📈 Order Success vs Failure")
    
    success_rate = round((success_pos / total_pos) * 100, 2) if total_pos > 0 else 0
    
    col1, col2 = st.columns([2, 1])
    
    # -----------------------------
    # Donut Chart
    # -----------------------------
    with col1:
        fig, ax = plt.subplots()
    
        ax.pie(
            [success_pos, failed_pos],
            labels=["Success", "Failure"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(width=0.35)  # donut effect
        )
    
        ax.axis("equal")
    
        st.pyplot(fig)
    
    
    # -----------------------------
    # Success Rate Metric
    # -----------------------------
    with col2:
        st.metric(
            label="✅ Success Rate",
            value=f"{success_rate} %",
            delta=f"{success_pos} / {total_pos} Orders" 
        )

    # ===================================================
    # Download Report Section
    # ===================================================

    st.markdown("""
<style>
div.stDownloadButton > button {
    background-color: #2ca02c;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    border: none;
}

div.stDownloadButton > button:hover {
    background-color: #155a8a;
    color: white;
}
</style>
""", unsafe_allow_html=True)
    
    import io
    import pandas as pd
    
    st.subheader("⬇️ Download Report")
    
    # -----------------------------
    # Prepare Summary Data
    # -----------------------------
    success_rate = round((success_pos / total_pos) * 100, 2) if total_pos > 0 else 0
    
    summary_df = pd.DataFrame({
        "Metric": [
            "Total Orders",
            "Successful Orders",
            "Failed Orders",
            "Success Rate (%)",
            "Avg Processing Time (min)"
        ],
        "Value": [
            total_pos,
            success_pos,
            failed_pos,
            success_rate,
            processing_time
        ]
    })
    
    # -----------------------------
    # Create Excel in memory
    # -----------------------------
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Raw_Data", index=False)
    
    buffer.seek(0)
    
    # -----------------------------
    # Download Button
    # -----------------------------
    st.download_button(
        label="📥 Download Operational Report (Excel)",
        data=buffer,
        file_name="Operational_Report.xlsx",
        mime="application/vnd.ms-excel"
    )


# ===================================================
# PAGE 2 – DQ SCORE DISTRIBUTION
# ===================================================
elif page == "📈 DQ Score Distribution":

    st.header("📈 DQ Score Distribution")
    # ---------------------------------------------------
    # Sidebar – Incoming EDI Order
    # ---------------------------------------------------
    st.sidebar.header("🧾 Incoming EDI Orders")
    
    missing = st.sidebar.slider("Missing Mandatory Fields", 0, 3, 0)
    invalid_ref = st.sidebar.slider("Invalid Reference Count", 0, 2, 0)
    format_err = st.sidebar.slider("Format Error Count", 0, 2, 0)
    partner_err = st.sidebar.slider("Partner Rule Violations", 0, 2, 0)
    order_lines = st.sidebar.slider("Number of Order Lines", 1, 20, 5)
    
    # ---------------------------------------------------
    # DQ Score Calculation
    # ---------------------------------------------------
    dq_score = 100 - (
        missing * 15 +
        invalid_ref * 20 +
        format_err * 5 +
        partner_err * 10
    )
    dq_score = max(dq_score, 0)
    
    # DQ Band
    if dq_score >= 80:
        dq_band = "🟢 Green"
    elif dq_score >= 50:
        dq_band = "🟠 Amber"
    else:
        dq_band = "🔴 Red"


        # -----------------------------
    # Incoming Order Snapshot
    # -----------------------------
    st.markdown("### 📍 Incoming EDI Order – Data Quality Snapshot")

    st.info(
        f"""
        **Current Incoming Order DQ Score:** {dq_score}  
        **DQ Band:** {dq_band}
        """
    )

    # -----------------------------
    # Data Quality Summary (OLD VIEW)
    # -----------------------------
    st.subheader("🧮 Data Quality Summary")

    dq1, dq2, dq3, dq4 = st.columns(4)
    dq1.metric("DQ Score", dq_score)
    dq2.metric("DQ Band", dq_band)
    dq3.metric("Missing Fields", missing)
    dq4.metric("Invalid References", invalid_ref)
    
    # ---------------------------------
    # Layout: Chart + Interpretation Table
    # ---------------------------------

    st.subheader("📈 Data Quality Score Distribution")
    
    col_chart, col_table = st.columns([2, 1])

    
    # ========= RIGHT: STATIC INTERPRETATION TABLE =========
    with col_table:

        st.markdown("### 📘 DQ Score Interpretation")

        dq_table = pd.DataFrame({
            "DQ Score Range": ["80 – 100", "50 – 79", "Below 50"],
            "Quality Band": ["High (Green)", "Medium (Amber)", "Low (Red)"],
            "Interpretation": [
                "Order is reliable and low risk",
                "Order requires attention",
                "Order is high risk and likely to fail"
            ]
        })

        st.table(dq_table)

        st.caption("Table: DQ Score Bands and Interpretation")
# ========= LEFT: DQ DISTRIBUTION CHART =========
    with col_chart:

        band_counts = pd.cut(
            data["dq_score"],
            bins=[0, 50, 80, 100],
            labels=["Low (Red)", "Medium (Amber)", "High (Green)"]
        ).value_counts().reindex(
            ["Low (Red)", "Medium (Amber)", "High (Green)"]
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(
            band_counts.index,
            band_counts.values,
            color=["red", "orange", "green"],
            alpha=0.75
        )

        ax.set_xlabel("DQ Quality Band")
        ax.set_ylabel("Number of Orders")
        ax.set_title("Overall DQ Score Distribution")

        st.pyplot(fig)

        st.info(
            f"""
            **Incoming Order DQ Score:** {dq_score}  
            **Quality Band:** {dq_band}
            """
        )


# ===================================================
# PAGE 3 – FAILURE RISK LEVELS
# ===================================================
elif page == "🚨 Failure Risk Levels":

    st.subheader("🚨 Failure Risk Levels")
    
    data["Risk Level"] = pd.cut(
        data["dq_score"], bins=[0,50,80,100],
        labels=["High","Medium","Low"]
    )

    st.bar_chart(data["Risk Level"].value_counts())

    st.dataframe(
        data[["po_id","dq_score","Risk Level","order_failed"]]
        .rename(columns={
            "po_id":"PO ID",
            "dq_score":"DQ Score",
            "order_failed":"Actual Failure"
        }),
        use_container_width=True
    )

    # ---------------------------------------------------
    # PREDICTION RESULTS
    # ---------------------------------------------------
    
    st.subheader("🔮 Predictive Results")
    
    # ---------------------------------------------------
    # Train models
    # ---------------------------------------------------
    rf_model, xgb_model, time_model, rf_acc, xgb_acc = train_models(data)
    
    st.caption(f"Model Accuracy → RF: {rf_acc:.2f} | XGB: {xgb_acc:.2f}")
    
    # ---------------------------------------------------
    # Sidebar Inputs for Prediction
    # ---------------------------------------------------
    st.sidebar.header("🔮 Predict New Order")
    
    missing = st.sidebar.slider("Missing Fields", 0, 3, 0, key="f1")
    invalid_ref = st.sidebar.slider("Invalid Ref", 0, 2, 0, key="f2")
    format_err = st.sidebar.slider("Format Errors", 0, 2, 0, key="f3")
    partner_err = st.sidebar.slider("Partner Errors", 0, 2, 0, key="f4")
    order_lines = st.sidebar.slider("Order Lines", 1, 20, 5, key="f5")
    
    dq_score = 100 - (
        missing * 15 +
        invalid_ref * 20 +
        format_err * 5 +
        partner_err * 10
    )
    dq_score = max(dq_score, 0)
    
    # ---------------------------------------------------
    # Prepare prediction row
    # ---------------------------------------------------
    input_df = pd.DataFrame([{
        "dq_score": dq_score,
        "missing": missing,
        "invalid_ref": invalid_ref,
        "format_err": format_err,
        "partner_err": partner_err,
        "order_lines": order_lines
    }])
    
    # ---------------------------------------------------
    # Cached Predictions (FAST)
    # ---------------------------------------------------
    fail_prob = xgb_model.predict_proba(input_df)[0][1]
    pred_time = time_model.predict(input_df)[0]

    
    # ---------------------------------------------------
    # PREDICTION RESULTS UI (unchanged)
    # ---------------------------------------------------
    r1, r2 = st.columns(2)
    
    r1.metric("Failure Probability", f"{round(fail_prob*100, 2)}%")
    r2.metric("Predicted Processing Time", f"{round(pred_time, 2)} min")
    
    if fail_prob > 0.7:
        st.error("🔴 High Risk → Manual Review / Quarantine")
    elif fail_prob > 0.4:
        st.warning("🟠 Medium Risk → Monitor Closely")
    else:
        st.success("🟢 Low Risk → Auto Processing")



# ===================================================
# PAGE 4 – PROCESSING TIME & SLA TRENDS
# ===================================================
elif page == "⏱️ Processing Time & SLA Trends":

    st.subheader("⏱️ Processing Time & SLA Trends")

    fig, ax = plt.subplots()
    ax.scatter(data["dq_score"], data["processing_time_min"],
               c=data["order_failed"].map({0:"green",1:"red"}), alpha=0.6)
    ax.set_xlabel("DQ Score")
    ax.set_ylabel("Processing Time (min)")
    st.pyplot(fig)

# ===================================================
# PAGE 5 – DATA LAB
# ===================================================
elif page == "🧪 Data Lab (CSV / Synthetic)":

    st.subheader("🧪 Data Lab")

    uploaded = st.file_uploader("Upload CSV for testing", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(), use_container_width=True)

    n = st.slider("Generate Synthetic Orders", 100, 2000, 500)
    if st.button("Generate Synthetic Data"):
        gen = pd.DataFrame({
            "PO ID": range(9001, 9001+n),
            "DQ Score": np.random.randint(30,100,n)
        })
        st.dataframe(gen.head(), use_container_width=True)

# ===================================================
# PAGE 6 – ABOUT
# ===================================================
elif page == "ℹ️ About Project":

    st.markdown("""
    # 🏢 EDI Control Tower  
    ### Predictive Risk • Data Quality • Processing Time Intelligence
    
    ---
    
    ## 📌 Overview
    **EDI Control Tower** is an intelligent operations dashboard designed to monitor, analyze, and predict the reliability of Electronic Data Interchange (EDI) order processing.  
    It combines **Data Quality scoring, Machine Learning models, and real-time analytics** to proactively detect failures, estimate processing delays, and improve SLA compliance.
    
    The platform helps operations and business teams move from **reactive issue handling → proactive risk prevention**.
    
    ---
    
    ## 🎯 Objectives
    - Improve order processing reliability  
    - Reduce failed transactions and rework  
    - Predict processing delays in advance  
    - Enable faster operational decisions  
    - Provide visibility into data quality issues  
    
    ---
    
    ## ⚙️ Key Capabilities
    ### 📊 Data Quality Scoring
    Automatically evaluates incoming EDI orders based on:
    - Missing mandatory fields  
    - Invalid references  
    - Format violations  
    - Partner rule breaches  
    
    Orders are classified into **Green / Amber / Red risk bands**.
    
    ### 🚨 Failure Risk Prediction
    Machine Learning models estimate the **probability of order failure** before processing, enabling:
    - Manual review
    - Exception handling
    - Risk mitigation
    
    ### ⏱ Processing Time Estimation
    Predicts expected processing duration to:
    - Identify SLA breaches
    - Plan workloads
    - Optimize throughput
    
    ### 📈 Operational Insights
    Interactive dashboards provide:
    - Risk distribution
    - DQ score trends
    - SLA performance
    - Predictive intelligence
    
    ---
    
    ## 🤖 Technology Stack
    - **Frontend:** Streamlit  
    - **Data Processing:** Pandas, NumPy  
    - **Machine Learning:** Random Forest, XGBoost  
    - **Visualization:** Matplotlib  
    - **Language:** Python  
    
    ---
    
    ## 🧠 Modeling Approach
    - Classification models → Failure prediction  
    - Regression models → Processing time estimation  
    - Synthetic dataset used for simulation and demonstration  
    - Designed to be extendable to real EDI transaction logs  
    
    ---
    
    ## 💼 Business Value
    The Control Tower enables organizations to:
    - Reduce operational costs  
    - Prevent order failures  
    - Improve partner experience  
    - Achieve better SLA compliance  
    - Enhance decision-making with predictive insights  
    
    ---
    
    ## 🎓 Academic Context
    **Project Title:** Predictive Modeling and Data Quality Assurance for EDI Order Processing  
    **Program:** BITS PILANI WILP – Final Semester  
    **Created By** Ajitabh Tiwari  
    This implementation serves as a proof-of-concept demonstrating how **AI/ML techniques can be integrated into enterprise EDI workflows** to deliver measurable operational improvements.
    
    ---
    
    ### ⚠️ Note
    Synthetic data is used for demonstration purposes only.
    """)

