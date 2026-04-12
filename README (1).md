📌 SAP Fraud Detection & Predictive Analytics Platform
End‑to‑End Enterprise Architecture | Azure Databricks | SAP S/4HANA | LangChain | Streamlit | SAC | Datasphere
🚀 Overview
This project is a full Fraud Detection & Predictive Intelligence Platform built using:
1) SAP S/4HANA tables (LFA1, LFBK, RBKP, BSIK, BSAK)
2) Azure Databricks (ML + Jobs + Pipelines)
3) XGBoost + Bayesian Hyperparameter Optimization
4) Isolation Forest (Unsupervised Anomaly Detection)
5) Logistic Regression (Process Integrity Models)
6) SHAP Explainability
7) ARIMA Time Series Forecasting (Cash Outflow)
8) LangChain (RAG‑based Fraud Q&A)
9) Streamlit Web App (Interactive Dashboard + Chatbot)
10) SAP Datasphere + SAP Analytics Cloud (Reporting Layer)
This is a complete end‑to‑end enterprise solution, covering ingestion → ML → explainability → forecasting → reporting → chatbot.
🧩 Architecture
SAP S/4HANA (LFA1, LFBK, RBKP, BSIK, BSAK)
        │
        ▼
Python Ingestion (VS Code)
        │
        ▼
Unified Fraud Dataset (7M+ rows)
        │
        ▼
Azure Databricks
    ├── Feature Engineering
    ├── XGBoost + BayesSearchCV
    ├── Isolation Forest
    ├── Logistic Regression (3 risk models)
    ├── SHAP Explainability
    ├── ARIMA Forecasting
    └── Job Orchestration Pipeline
        │
        ▼
Unity Catalog Delta Tables
        │
        ├── Streamlit App (Dashboard + LangChain Chatbot)
        └── SAP Datasphere → SAP Analytics Cloud (SAC Reports)

📥 Data Sources
1) SAP S/4HANA Tables
     A) LFA1 – Vendor Master
     B) LFBK – Vendor Bank Details
     C) RBKP – Invoice Header
     D) BSIK – Open Items
     E) BSAK – Cleared Items
2) External Datasets
     A) IEEE Fraud Dataset
     B) PaySim Financial Transactions
     C) UK Government Payments (Vendor Enrichment)
All datasets were aligned, standardized, and unified into a single fraud dataset.

🧠 Machine Learning Models

1️⃣ XGBoost (Supervised Fraud Detection)
     A) Bayesian hyperparameter optimization using BayesSearchCV
     B) Reduced training time from 1 hour → 15 minutes
     C) Achieved AUC = 0.909
     D) Feature importance + SHAP explanations

2️⃣ Isolation Forest (Unsupervised Anomaly Detection)
     A) Multi‑scenario contamination strategy
     B) Flags:
          a) Standard
          b) Expense
         c) Low‑Volume Vendors
     C) Identifies structuring, invoice flooding, and rare events

3️⃣ Logistic Regression (Process Integrity Models)
     A) Discount Capture Model
     B) Late Payment Risk Model
     C) Manual vs Automated Entry Model

4️⃣ ARIMA Forecasting
     A) Daily cash outflow forecasting
     B) Vendor‑level temporal analysis

🔍 Explainability (SHAP)
1) Global feature importance
2) Local explanations for top anomalies
3) Waterfall plots for anomaly drivers
4) Business‑friendly insights
⚙️ Azure Databricks Pipeline

Jobs Created
1) 00_setup_and_data_load
2) 01_xgboost_fraud_classifier
3) 02_isolation_forest_anomaly
4) 03_risk_scoring_and_logistic_layers
5) 04_time_series_forecasting
6) 05_orchestrator

Pipeline Features
1) Auto‑triggered workflows
2) Cluster auto‑termination
3) Modular notebook design
4) Unity Catalog table outputs
💬 LangChain Fraud Intelligence Chatbot

Built using:
1) LangChain DataFrame Agent
2) OpenAI Chat Model
3) RAG‑style SQL + DataFrame retrieval
4) Streamlit UI

Capabilities:
1) Vendor risk Q&A
2) Fraud pattern explanations
3) Forecast interpretation
4) Scenario‑based insights

🌐 Streamlit Web Application
Features:
1) High‑risk invoice explorer
2) Vendor intelligence dashboard
3) Anomaly visualizations
4) Forecast charts
5) Integrated LangChain chatbot

📊 SAP Datasphere + SAC Reporting
1) Delta tables exported to Datasphere
2) SAC dashboards built for:
3) Vendor risk
4) Fraud probability
5) Anomaly distribution
6) Cash flow forecasting

🏁 End‑to‑End Outcome
This project demonstrates:
1) SAP data engineering
2) ML model development
3) Explainability
4) Forecasting
5) Cloud orchestration
6) RAG‑based chatbot
7) Enterprise reporting
8) A complete Fraud Analytics Platform.           
