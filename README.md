# 🏢 EDI Control Tower(edi-predictive-dashboard)
edi-predictive-dashboard
🏢 EDI Control Tower
Predictive Risk • Data Quality • Processing Time Intelligence
📌 Project Overview

EDI Control Tower is an intelligent analytics dashboard built using Streamlit + Machine Learning to monitor and predict the reliability of Electronic Data Interchange (EDI) order processing.

The system transforms raw operational data into:

Data Quality scores

Failure probability predictions

Processing time forecasts

SLA risk indicators

Actionable operational insights

It enables organizations to move from:

Reactive issue handling → Proactive risk prevention

🎯 Problem Statement

Traditional EDI processing systems:

detect failures after they occur

lack early warning signals

have limited visibility into data quality

cause SLA breaches and manual rework

Goal

Build a predictive dashboard that:

✅ detects bad orders early
✅ predicts failure risk
✅ estimates processing delays
✅ provides operational control tower visibility


🏗 System Architecture
User (Browser)
      ↓
Streamlit UI
      ↓
Data Processing (Pandas / NumPy)
      ↓
ML Models (RF + XGBoost)
      ↓
Predictions & Visualizations

🧩 Technology Stack
| Layer      | Technology              |
| ---------- | ----------------------- |
| Frontend   | Streamlit               |
| Language   | Python                  |
| Data       | Pandas, NumPy           |
| ML         | RandomForest, XGBoost   |
| Charts     | Matplotlib              |
| Deployment | Streamlit Cloud / Local |

⚙️ Application Flow (High Level)

Generate or load dataset

Compute Data Quality metrics

Train ML models (cached)

User selects dashboard page

Predictions generated in real time

Insights displayed visually
