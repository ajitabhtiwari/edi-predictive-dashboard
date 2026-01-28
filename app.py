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
st.set_page_config(page_title="EDI Predictive Analytics Dashboard", layout="wide")

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
st.title("🚀 EDI Predictive Analytics Dashboard")
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
# Sidebar – Incoming EDI Order
# ---------------------------------------------------
st.sidebar.header("🧾 Incoming EDI Order")

missing = st.sidebar.slider("Missing Mandatory Fields", 0, 3, 0)
invalid_ref = st.sidebar.slider("Invalid Reference Count", 0, 2, 0)
format_err = st.sidebar.slider("Format Error Count", 0, 2, 0)
partner_err = st.sidebar.slider("Partner Rule Violations", 0, 2, 0)
order_lines = st.sidebar.slider("Number of Order Lines", 1, 20, 5)

# ---------------------------------------------------
# DQ Score
# ---------------------------------------------------
dq_score = max(100 - (missing*15 + invalid_ref*20 + format_err*5 + partner_err*10), 0)

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total POs", len(data))
    c2.metric("Successful Orders", len(data) - data.order_failed.sum())
    c3.metric("Failed Orders", data.order_failed.sum())
    c4.metric("Avg Processing Time (min)", round(data.processing_time_min.mean(),2))

# ===================================================
# PAGE 2 – DQ SCORE DISTRIBUTION
# ===================================================
elif page == "📈 DQ Score Distribution":

    st.subheader("📈 DQ Score Distribution")

    band_counts = pd.cut(
        data["dq_score"], bins=[0,50,80,100],
        labels=["Red","Amber","Green"]
    ).value_counts().reindex(["Red","Amber","Green"])

    fig, ax = plt.subplots()
    ax.bar(band_counts.index, band_counts.values, color=["red","orange","green"])
    ax.set_xlabel("DQ Band")
    ax.set_ylabel("Number of Orders")
    st.pyplot(fig)

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
**Project Title:** Predictive Modeling and Data Quality Assurance for EDI Order Processing  
**Program:** BITS WILP – Final Semester  

This dashboard demonstrates how Data Quality scoring and predictive analytics
can be applied to improve EDI order processing reliability and SLA compliance.
""")

st.caption("Synthetic data used for academic demonstration only.")
