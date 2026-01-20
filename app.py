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
st.markdown(
    """
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
        cursor: pointer;
    }

    .kpi:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 12px 24px rgba(0,0,0,0.35);
        filter: brightness(1.05);
    }

    .kpi h2 {
        font-size: 36px;
        margin: 0;
        font-weight: bold;
    }

    .kpi p {
        font-size: 15px;
        margin: 6px 0 0 0;
        opacity: 0.9;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True
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

@st.cache_resource
def train_models(data, features):

    X = data[features]
    y_fail = data["order_failed"]
    y_time = data["processing_time_min"]

    X_train, X_test, y_fail_train, y_fail_test = train_test_split(
        X, y_fail, test_size=0.3, random_state=42
    )

    _, _, y_time_train, y_time_test = train_test_split(
        X, y_time, test_size=0.3, random_state=42
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train, y_fail_train)

    xgb_model = XGBClassifier(
        n_estimators=120,
        learning_rate=0.1,
        max_depth=4,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_fail_train)

    time_model = XGBRegressor(
        n_estimators=120,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        verbosity=0
    )
    time_model.fit(X_train, y_time_train)

    rf_accuracy = accuracy_score(y_fail_test, rf_model.predict(X_test))
    xgb_accuracy = accuracy_score(y_fail_test, xgb_model.predict(X_test))

    return rf_model, xgb_model, time_model, rf_accuracy, xgb_accuracy

with st.spinner("⏳ Training predictive models (one-time setup)..."):
    rf_model, xgb_model, time_model, rf_accuracy, xgb_accuracy = train_models(
        data, features
    )

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

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi" style="background:#2ca02c;">
        <h2>{total_pos}</h2>
        <p>📦 Total POs</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi" style="background:#b32400;">
        <h2>{failed_pos}</h2>
        <p>❌ Failed Orders</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi" style="background:#6699cc;">
        <h2>{round(rf_accuracy*100,2)}%</h2>
        <p>🌲 RF Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi" style="background:#80ff80;">
        <h2>{round(xgb_accuracy*100,2)}%</h2>
        <p>⚡ XGBoost Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

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
    st.subheader("📊 DQ Score Band Distribution")
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
    # ax.set_title("DQ Score Band Distribution")

    st.pyplot(fig)

with col2:
    st.subheader("⏱️ Processing Time vs Data Quality Score")
    # Prepare data
    plot_df = data.copy()
    plot_df["Status"] = np.where(plot_df["order_failed"] == 1, "Failed", "Successful")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plot_df["Status"].map({
        "Successful": "#2ca02c",
        "Failed": "#d62728"
    })

    sizes = plot_df["order_lines"] * 8  # bubble size

    scatter = ax.scatter(
        plot_df["dq_score"],
        plot_df["processing_time_min"],
        s=sizes,
        c=colors,
        alpha=0.6
    )

    # Trend line
    z = np.polyfit(plot_df["dq_score"], plot_df["processing_time_min"], 1)
    p = np.poly1d(z)
    ax.plot(
        plot_df["dq_score"],
        p(plot_df["dq_score"]),
        linestyle="--",
        color="black",
        linewidth=2,
        label="Trend"
    )

    ax.set_xlabel("Data Quality Score")
    ax.set_ylabel("Processing Time (minutes)")
    ax.set_title("Impact of Data Quality on Processing Time")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Successful',
               markerfacecolor='#2ca02c', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Failed',
               markerfacecolor='#d62728', markersize=10),
    ]

    ax.legend(handles=legend_elements, loc="upper right")

    st.pyplot(fig)
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
