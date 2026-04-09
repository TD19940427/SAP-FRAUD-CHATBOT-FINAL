import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="SAP Fraud Detection Chatbot",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 SAP Fraud Detection Intelligence Chatbot")
st.markdown("Ask me anything about vendor risk scores, fraud patterns, invoices, and forecasts!")

# Sidebar
with st.sidebar:
    st.header("📊 Data Overview")
    
    st.markdown("### 💡 Example Questions:")
    st.markdown("""
    - What's the risk score for vendor V10848?
    - Show me top 10 high-risk invoices
    - Which vendors have manual entry patterns?
    - What's the average fraud probability?
    - Forecast vendor V10848's behavior
    - Which vendors have late payment risk?
    - Show anomalies with amount > 50000
    - What's the total invoice count?
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load data with caching
@st.cache_data
def load_data():
    """Load all fraud detection datasets"""
    try:
        # Load main dataset
        df = pd.read_csv("sap_invoice_risk_master_SMALL.csv")
        df['date'] = pd.to_datetime(df['date'])
        
        # Load vendor summary
        vendor_summary = pd.read_csv("vendor_intelligence_summary_GITHUB.csv")
        
        # Load V10848 forecast
        df_v10848 = pd.read_csv("v10848_temporal_analysis_GITHUB.csv")
        if 'ds' in df_v10848.columns:
            df_v10848['ds'] = pd.to_datetime(df_v10848['ds'])
        elif 'date' in df_v10848.columns:
            df_v10848['date'] = pd.to_datetime(df_v10848['date'])
        
        return df, vendor_summary, df_v10848
    except FileNotFoundError as e:
        st.error(f"❌ Error: Data file not found - {e}")
        st.info("Make sure these files are in the same directory: sap_invoice_risk_master_SMALL.csv, vendor_intelligence_summary_GITHUB.csv, v10848_temporal_analysis_GITHUB.csv")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None

# Load the data
with st.spinner("📥 Loading fraud detection data..."):
    df, vendor_summary, df_v10848 = load_data()

if df is not None:
    with st.sidebar:
        st.markdown("### 📈 Dataset Stats:")
        st.metric("Total Invoices", f"{len(df):,}")
        st.metric("Total Vendors", f"{df['vendor_id'].nunique():,}")
        if 'manual_pattern_score' in df.columns:
            st.metric("Avg Manual Score", f"{df['manual_pattern_score'].mean():.3f}")
        st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
        st.success("✅ Data loaded!")
else:
    st.error("❌ Cannot proceed without data files.")
    st.stop()

# ============================================================
# SMART QUERY SYSTEM (No AI needed!)
# ============================================================

def query_dataframes(question: str, df, vendor_summary, df_v10848) -> str:
    """Rule-based query system for fraud detection data"""
    
    question_lower = question.lower()
    
    # Vendor V10848 specific queries
    if "v10848" in question_lower:
        vendor_data = df[df['vendor_id'] == 'V10848']
        
        if "risk" in question_lower or "score" in question_lower:
            if len(vendor_data) == 0:
                return "No data found for vendor V10848 in the dataset."
            avg_manual = vendor_data['manual_pattern_score'].mean()
            avg_late = vendor_data['late_payment_risk'].mean()
            return f"""**Vendor V10848 Risk Profile:**
- Manual Pattern Score: {avg_manual:.3f}
- Late Payment Risk: {avg_late:.3f}
- Total Invoices: {len(vendor_data):,}
- Average Amount: ${vendor_data['amount'].mean():,.2f}
- Date Range: {vendor_data['date'].min().date()} to {vendor_data['date'].max().date()}"""
        
        elif "forecast" in question_lower or "prediction" in question_lower:
            if len(df_v10848) > 0:
                latest = df_v10848.iloc[-1]
                return f"""**V10848 Forecast (ARIMA Model):**
- Latest Actual: {latest['actual']}
- Latest Forecast: {latest['forecast']:.2f}
- Residual: {latest['residual']:.2f}
- Behavioral Anomaly: {'Yes' if latest['behavioral_anomaly'] == 1 else 'No'}
- Total Forecast Periods: {len(df_v10848):,}"""
            else:
                return "No forecast data available for V10848."
        
        else:
            return f"""**Vendor V10848 Overview:**
- Total Invoices: {len(vendor_data):,}
- Average Amount: ${vendor_data['amount'].mean():,.2f}
- Total Amount: ${vendor_data['amount'].sum():,.2f}"""
    
    # High-risk queries
    elif "high risk" in question_lower or ("top" in question_lower and "invoice" in question_lower):
        top_invoices = df.nlargest(10, 'amount')[['vendor_id', 'amount', 'date', 'manual_pattern_score', 'late_payment_risk']]
        result = "**Top 10 Highest Risk Invoices:**\n\n"
        for idx, row in top_invoices.iterrows():
            result += f"- **{row['vendor_id']}**: ${row['amount']:,.2f} on {row['date'].date()}\n"
            result += f"  - Manual Score: {row['manual_pattern_score']:.3f}, Late Risk: {row['late_payment_risk']:.3f}\n"
        return result
    
    # Manual entry pattern queries
    elif "manual" in question_lower and "pattern" in question_lower:
        top_manual = df.groupby('vendor_id')['manual_pattern_score'].mean().nlargest(10)
        result = "**Top 10 Vendors by Manual Entry Pattern Score:**\n\n"
        for vendor, score in top_manual.items():
            count = len(df[df['vendor_id'] == vendor])
            result += f"- **{vendor}**: {score:.3f} ({count:,} invoices)\n"
        return result
    
    # Late payment risk queries
    elif "late payment" in question_lower:
        if ">" in question_lower or "above" in question_lower:
            try:
                threshold = 0.8
                if ">" in question_lower:
                    threshold = float(question_lower.split(">")[1].strip().split()[0])
                elif any(word in question_lower for word in ["0.7", "0.8", "0.9"]):
                    for word in ["0.9", "0.8", "0.7"]:
                        if word in question_lower:
                            threshold = float(word)
                            break
            except:
                threshold = 0.8
            
            high_late = df[df['late_payment_risk'] > threshold].groupby('vendor_id').size().nlargest(10)
            result = f"**Top 10 Vendors with Late Payment Risk > {threshold}:**\n\n"
            for vendor, count in high_late.items():
                avg_risk = df[df['vendor_id'] == vendor]['late_payment_risk'].mean()
                result += f"- **{vendor}**: {count:,} invoices (avg risk: {avg_risk:.3f})\n"
            return result
        else:
            avg_late = df['late_payment_risk'].mean()
            high_late_count = len(df[df['late_payment_risk'] > 0.7])
            return f"""**Late Payment Risk Analysis:**
- Average late payment risk: {avg_late:.3f}
- Invoices with high risk (>0.7): {high_late_count:,} ({high_late_count/len(df)*100:.1f}%)"""
    
    # Anomaly queries
    elif "anomal" in question_lower:
        if 'anomaly_flag' in df.columns:
            anomalies = df[df['anomaly_flag'] == 1]
            
            if "amount" in question_lower and ">" in question_lower:
                try:
                    threshold = float(question_lower.split(">")[1].strip().split()[0])
                    filtered = anomalies[anomalies['amount'] > threshold]
                    return f"""**Anomalies with Amount > ${threshold:,.0f}:**
- Count: {len(filtered):,}
- Average Amount: ${filtered['amount'].mean():,.2f}
- Total Amount: ${filtered['amount'].sum():,.2f}
- Top Vendor: {filtered.groupby('vendor_id').size().idxmax() if len(filtered) > 0 else 'N/A'}"""
                except:
                    pass
            
            return f"""**Anomaly Analysis:**
- Total Anomalies: {len(anomalies):,} ({len(anomalies)/len(df)*100:.1f}% of invoices)
- Average Amount: ${anomalies['amount'].mean():,.2f}
- Top 3 Vendors: {', '.join(anomalies.groupby('vendor_id').size().nlargest(3).index.tolist())}"""
        else:
            return "Anomaly flag column not found in dataset."
    
    # Statistical queries
    elif "average" in question_lower or "mean" in question_lower:
        if "fraud probability" in question_lower and 'fraud_probability' in df.columns:
            return f"Average fraud probability across all invoices: {df['fraud_probability'].mean():.3f}"
        elif "amount" in question_lower:
            return f"""**Invoice Amount Statistics:**
- Average: ${df['amount'].mean():,.2f}
- Median: ${df['amount'].median():,.2f}
- Max: ${df['amount'].max():,.2f}
- Min: ${df['amount'].min():,.2f}"""
    
    # Total/count queries
    elif "total" in question_lower and "invoice" in question_lower:
        return f"""**Dataset Overview:**
- Total Invoices: {len(df):,}
- Unique Vendors: {df['vendor_id'].nunique():,}
- Date Range: {df['date'].min().date()} to {df['date'].max().date()}
- Total Amount: ${df['amount'].sum():,.2f}"""
    
    # Vendor statistics
    elif "vendor" in question_lower and ("how many" in question_lower or "count" in question_lower):
        return f"The dataset contains **{df['vendor_id'].nunique():,} unique vendors** with {len(df):,} total invoices."
    
    # Default response - show capabilities
    else:
        return f"""**I can help you analyze:**

📊 **Dataset Summary:**
- Total Invoices: {len(df):,}
- Unique Vendors: {df['vendor_id'].nunique():,}
- Average Amount: ${df['amount'].mean():,.2f}
- Date Range: {df['date'].min().date()} to {df['date'].max().date()}

**Try asking:**
- "What's the risk score for vendor V10848?"
- "Show me top 10 high-risk invoices"
- "Which vendors have manual entry patterns?"
- "Show anomalies with amount > 50000"
- "What's the average fraud probability?"
"""

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about the fraud detection data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            response = query_dataframes(prompt, df, vendor_summary, df_v10848)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with 🐍 Python | 🎈 Streamlit<br>
    SAP Fraud Detection Intelligence System | 100% FREE - No API Keys Needed!
</div>
""", unsafe_allow_html=True)
