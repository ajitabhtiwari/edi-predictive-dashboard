import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="EDI Predictive Analytics Dashboard",
    layout="wide"
)

st.title("🚀 EDI Predictive Analytics Dashboard")
st.markdown("""
**Scope:**  
Data Quality Scoring • Failure Risk Prediction • Processing Time Estimation  
*(Synthetic data – academic demonstration)*
""")

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

# ---------------------------------------------------
# Synthetic Dataset (Final Phase)
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

# Failure Logic
data["order_failed"] = (
    (data["missing"] > 1) |
    (data["invalid_ref"] > 0)
).astype(int)

# Processing Time Logic
data["processing_time_min"] = (
    5 +
    (100 - data["dq_score"]) * 0.3 +
    data["order_lines"] * 0.8 +
    data["order_failed"] * 10
).round(2)

# ---------------------------------------------------
# Feature & Target Sets
# ---------------------------------------------------
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

# ---------------------------------------------------
# MODELS – CLASSIFICATION
# ---------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
rf_model.fit(X_train, y_fail_train)

rf_accuracy = accuracy_score(
    y_fail_test,
    rf_model.predict(X_test)
)

xgb_model = XGBClassifier(
    n_estimators=120,
    learning_rate=0.1,
    max_depth=4,
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_fail_train)

xgb_accuracy = accuracy_score(
    y_fail_test,
    xgb_model.predict(X_test)
)

# ---------------------------------------------------
# MODEL – PROCESSING TIME (REGRESSION)
# ---------------------------------------------------
time_model = XGBRegressor(
    n_estimators=120,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
time_model.fit(X_train, y_time_train)

# ---------------------------------------------------
# Prediction for Current Order
# ---------------------------------------------------
input_df = pd.DataFrame([[
    dq_score,
    missing,
    invalid_ref,
    format_err,
    partner_err,
    order_lines
]], columns=features)

fail_prob = xgb_model.predict_proba(input_df)[0][1]
pred_time = time_model.predict(input_df)[0]

# ---------------------------------------------------
# OPERATIONAL SUMMARY – KPI CARDS
# ---------------------------------------------------
st.subheader("📊 Operational Summary")

total_pos = len(data)
failed_pos = data["order_failed"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("📦 Total POs", total_pos)
k2.metric("❌ Failed Orders", failed_pos)
k3.metric("🌲 RF Accuracy", f"{round(rf_accuracy*100, 2)}%")
k4.metric("⚡ XGBoost Accuracy", f"{round(xgb_accuracy*100, 2)}%")

# ---------------------------------------------------
# DQ METRICS
# ---------------------------------------------------
st.subheader("🧮 Data Quality Summary")

dq1, dq2, dq3, dq4 = st.columns(4)
dq1.metric("DQ Score", dq_score)
dq2.metric("DQ Band", dq_band)
dq3.metric("Missing Fields", missing)
dq4.metric("Invalid References", invalid_ref)

# ---------------------------------------------------
# PREDICTION RESULTS
# ---------------------------------------------------
st.subheader("🔮 Predictive Results")

r1, r2 = st.columns(2)
r1.metric("Failure Probability", f"{round(fail_prob*100, 2)}%")
r2.metric("Predicted Processing Time", f"{round(pred_time,2)} min")

if fail_prob > 0.7:
    st.error("🔴 High Risk → Manual Review / Quarantine")
elif fail_prob > 0.4:
    st.warning("🟠 Medium Risk → Monitor Closely")
else:
    st.success("🟢 Low Risk → Auto Processing")

# ---------------------------------------------------
# FINAL PHASE CHARTS
# ---------------------------------------------------
st.subheader("📈 Advanced Analytics")

col1, col2 = st.columns(2)

with col1:
    st.write("**DQ Score Band Distribution 101**")

    band_counts = pd.cut(
        data["dq_score"],
        bins=[0, 50, 80, 100],
        labels=["Red", "Amber", "Green"]
    ).value_counts().reindex(["Red", "Amber", "Green"])

    colors = ["red", "orange", "green"]

    fig, ax = plt.subplots()
    ax.bar(band_counts.index, band_counts.values, color=colors)

    ax.set_xlabel("DQ Band")
    ax.set_ylabel("Number of Orders")
    ax.set_title("DQ Score Band Distribution")

    st.pyplot(fig)

with col2:
    st.write("**Processing Time vs DQ Score**")
    st.scatter_chart(
        data,
        x="dq_score",
        y="processing_time_min"
    )

# ---------------------------------------------------
# PO LEVEL TABLE
# ---------------------------------------------------
st.subheader("📋 PO-Level Outcomes")

data_view = data.copy()
data_view["DQ_Band"] = pd.cut(
    data_view["dq_score"],
    bins=[0,50,80,100],
    labels=["Red","Amber","Green"]
)
data_view["Risk"] = np.where(
    data_view["order_failed"] == 1, "High", "Low"
)

st.dataframe(
    data_view[[
        "po_id",
        "dq_score",
        "DQ_Band",
        "order_lines",
        "processing_time_min",
        "Risk"
    ]],
    use_container_width=True
)

st.caption("Synthetic data used for final semester academic demonstration.")
