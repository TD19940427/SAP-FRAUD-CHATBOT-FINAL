📌 SAP Fraud Detection & Predictive Analytics Platform End‑to‑End Enterprise Architecture | Azure Databricks | SAP S/4HANA | LangChain | Streamlit | SAC | Datasphere 🚀 Overview This project is a full Fraud Detection & Predictive Intelligence Platform built using:

SAP S/4HANA tables (LFA1, LFBK, RBKP, BSIK, BSAK)
Azure Databricks (ML + Jobs + Pipelines)
XGBoost + Bayesian Hyperparameter Optimization
Isolation Forest (Unsupervised Anomaly Detection)
Logistic Regression (Process Integrity Models)
SHAP Explainability
ARIMA Time Series Forecasting (Cash Outflow)
LangChain (RAG‑based Fraud Q&A)
Streamlit Web App (Interactive Dashboard + Chatbot)
SAP Datasphere + SAP Analytics Cloud (Reporting Layer) This is a complete end‑to‑end enterprise solution, covering ingestion → ML → explainability → forecasting → reporting → chatbot. 🧩 Architecture SAP S/4HANA (LFA1, LFBK, RBKP, BSIK, BSAK) │ ▼ Python Ingestion (VS Code) │ ▼ Unified Fraud Dataset (7M+ rows) │ ▼ Azure Databricks ├── Feature Engineering ├── XGBoost + BayesSearchCV ├── Isolation Forest ├── Logistic Regression (3 risk models) ├── SHAP Explainability ├── ARIMA Forecasting └── Job Orchestration Pipeline │ ▼ Unity Catalog Delta Tables │ ├── Streamlit App (Dashboard + LangChain Chatbot) └── SAP Datasphere → SAP Analytics Cloud (SAC Reports)
📥 Data Sources

SAP S/4HANA Tables A) LFA1 – Vendor Master B) LFBK – Vendor Bank Details C) RBKP – Invoice Header D) BSIK – Open Items E) BSAK – Cleared Items
External Datasets A) IEEE Fraud Dataset B) PaySim Financial Transactions C) UK Government Payments (Vendor Enrichment) All datasets were aligned, standardized, and unified into a single fraud dataset.
🧠 Machine Learning Models

1️⃣ XGBoost (Supervised Fraud Detection) A) Bayesian hyperparameter optimization using BayesSearchCV B) Reduced training time from 1 hour → 15 minutes C) Achieved AUC = 0.909 D) Feature importance + SHAP explanations

2️⃣ Isolation Forest (Unsupervised Anomaly Detection) A) Multi‑scenario contamination strategy B) Flags: a) Standard b) Expense c) Low‑Volume Vendors C) Identifies structuring, invoice flooding, and rare events

3️⃣ Logistic Regression (Process Integrity Models) A) Discount Capture Model B) Late Payment Risk Model C) Manual vs Automated Entry Model

4️⃣ ARIMA Forecasting A) Daily cash outflow forecasting B) Vendor‑level temporal analysis

🔍 Explainability (SHAP)

Global feature importance
Local explanations for top anomalies
Waterfall plots for anomaly drivers
Business‑friendly insights ⚙️ Azure Databricks Pipeline
Jobs Created

00_setup_and_data_load
01_xgboost_fraud_classifier
02_isolation_forest_anomaly
03_risk_scoring_and_logistic_layers
04_time_series_forecasting
05_orchestrator
Pipeline Features

Auto‑triggered workflows
Cluster auto‑termination
Modular notebook design
Unity Catalog table outputs 💬 LangChain Fraud Intelligence Chatbot
Built using:

LangChain DataFrame Agent
OpenAI Chat Model
RAG‑style SQL + DataFrame retrieval
Streamlit UI
Capabilities:

Vendor risk Q&A
Fraud pattern explanations
Forecast interpretation
Scenario‑based insights
🌐 Streamlit Web Application Features:

High‑risk invoice explorer
Vendor intelligence dashboard
Anomaly visualizations
Forecast charts
Integrated LangChain chatbot
📊 SAP Datasphere + SAC Reporting

Delta tables exported to Datasphere
SAC dashboards built for:
Vendor risk
Fraud probability
Anomaly distribution
Cash flow forecasting
🏁 End‑to‑End Outcome This project demonstrates:

SAP data engineering
ML model development
Explainability
Forecasting
Cloud orchestration
RAG‑based chatbot
Enterprise reporting
A complete Fraud Analytics Platform.
