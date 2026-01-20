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
st.set_page_config(
    page_title="EDI Predictive Analytics Dashboard",
    layout="wide"
)
# ---------------------------------------------------
# LOGO & TITLE
# ---------------------------------------------------
st.image("logo.png", width=160)
st.title("🚀 EDI Predictive Analytics Dashboard")
st.markdown("""
**Scope:**  
Data Quality Scoring • Failure Risk Prediction • Processing Time Estimation  
*(Synthetic data – academic demonstration)*
""")

# ---------------------------------------------------
# STYLES
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
    transition: all 0.3s ease-in-out;
}
.kpi:hover {
    transform: translateY(-8px) scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.image("logo.png", width=120)
st.sidebar.markdown("## Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard Overview",
        "🧪 Synthetic Data Generator",
        "📋 PO-Level Analysis",
        "ℹ️ About Project"
    ]
)

# ---------------------------------------------------
# COMMON DATA (Used Across Pages)
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

data["order_failed"] = (
    (data["missing"] > 1) |
    (data["invalid_ref"] > 0)
).astype(int)

data["processing_time_min"] = (
    5 +
    (100 - data["dq_score"]) * 0.3 +
    data["order_lines"] * 0.8 +
    data["order_failed"] * 10
).round(2)

features = [
    "dq_score","missing","invalid_ref",
    "format_err","partner_err","order_lines"
]

# ---------------------------------------------------
# MODEL TRAINING (CACHED)
# ---------------------------------------------------
@st.cache_resource
def train_models(data):
    X = data[features]
    y = data["order_failed"]
    y_time = data["processing_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=120, learning_rate=0.1,
        max_depth=4, eval_metric="logloss", verbosity=0
    )
    xgb.fit(X_train, y_train)

    time_model = XGBRegressor(
        n_estimators=120, learning_rate=0.1, max_depth=4, verbosity=0
    )
    time_model.fit(X_train, y_time)

    return (
        rf, xgb, time_model,
        accuracy_score(y_test, rf.predict(X_test)),
        accuracy_score(y_test, xgb.predict(X_test))
    )

rf_model, xgb_model, time_model, rf_acc, xgb_acc = train_models(data)

# ===================================================
# PAGE 1 – DASHBOARD OVERVIEW
# ===================================================
if page == "📊 Dashboard Overview":

    st.subheader("📊 Operational Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi' style='background:#2ca02c;'><h2>{len(data)}</h2><p>Total POs</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi' style='background:#b32400;'><h2>{data.order_failed.sum()}</h2><p>Failed Orders</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi' style='background:#6699cc;'><h2>{rf_acc*100:.2f}%</h2><p>RF Accuracy</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi' style='background:#80ff80;'><h2>{xgb_acc*100:.2f}%</h2><p>XGBoost Accuracy</p></div>", unsafe_allow_html=True)

    st.subheader("📊 DQ Score Band Distribution")

    band_counts = pd.cut(
        data["dq_score"], bins=[0,50,80,100],
        labels=["Red","Amber","Green"]
    ).value_counts().reindex(["Red","Amber","Green"])

    fig, ax = plt.subplots()
    ax.bar(band_counts.index, band_counts.values, color=["red","orange","green"])
    st.pyplot(fig)

# ===================================================
# PAGE 2 – SYNTHETIC DATA GENERATOR
# ===================================================
elif page == "🧪 Synthetic Data Generator":

    st.subheader("🧪 Synthetic EDI Data Generator")

    num = st.slider("Number of Orders", 100, 3000, 500)
    max_lines = st.slider("Max Order Lines", 5, 30, 20)

    if st.button("Generate Data"):
        gen = pd.DataFrame({
            "PO ID": range(10001, 10001 + num),
            "DQ Score": np.random.randint(30, 100, num),
            "Order Lines": np.random.randint(1, max_lines, num)
        })
        st.success("Synthetic data generated")
        st.dataframe(gen.head(20), use_container_width=True)
        st.download_button(
            "Download CSV",
            gen.to_csv(index=False),
            "synthetic_edi_data.csv"
        )

# ===================================================
# PAGE 3 – PO LEVEL ANALYSIS
# ===================================================
elif page == "📋 PO-Level Analysis":

    st.subheader("📋 PO-Level Outcomes")

    display_df = data[[
        "po_id","dq_score","order_lines",
        "processing_time_min","order_failed"
    ]].rename(columns={
        "po_id":"PO ID",
        "dq_score":"DQ Score",
        "order_lines":"Order Lines",
        "processing_time_min":"Processing Time (Min)",
        "order_failed":"Failed"
    })

    st.dataframe(display_df, use_container_width=True)

# ===================================================
# PAGE 4 – ABOUT
# ===================================================
elif page == "ℹ️ About Project":

    st.markdown("""
    **Project Title:** Predictive Modeling and Data Quality Assurance for EDI Order Processing  
    **Degree:** BITS WILP – Final Semester  
    **Author:** Ajitabh Tiwari  

    **Description:**  
    This project integrates rule-based data quality validation with
    machine learning models to predict EDI order failure risk and
    estimate processing time, enabling proactive decision-making.
    """)

st.caption("Synthetic data used for academic demonstration only.")
