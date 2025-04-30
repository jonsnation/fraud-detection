# import streamlit as st
# import pandas as pd
# import requests
# import plotly.express as px
# from streamlit_autorefresh import st_autorefresh

# # Auto-refresh every 5 seconds
# st_autorefresh(interval=5000, key="refresh")

# # App title
# st.title("Real-Time Fraud Detection Dashboard")

# # Fetch data from API
# API_URL = "https://your-api-url.com/transactions"
# try:
#     response = requests.get(API_URL)
#     response.raise_for_status()
#     data = response.json()
#     df = pd.DataFrame(data)
# except Exception as e:
#     st.error(f"Failed to fetch data from API: {e}")
#     st.stop()

# # Optional: Convert TransactionDT to datetime if needed
# # df['TransactionDT'] = pd.to_datetime(df['TransactionDT'], unit='s')

# # Sidebar filters
# st.sidebar.header("Filter")
# num_rows = st.sidebar.slider("Rows to show", 5, 100, 20)

# # Summary statistics
# st.subheader("Summary Stats")
# col1, col2, col3 = st.columns(3)
# col1.metric("Total Transactions", len(df))
# col2.metric("Fraudulent", int(df['isFraud'].sum()))
# col3.metric("Average Amount", f"${df['TransactionAmt'].mean():.2f}")

# # Show table
# st.subheader("Live Transaction Table")
# st.dataframe(df.tail(num_rows).sort_values("TransactionDT", ascending=False), use_container_width=True)

# # Fraud trend
# st.subheader("Fraud Detection Trend")
# fraud_trend = df.groupby("TransactionDT")["isFraud"].sum().reset_index()
# fig_trend = px.line(fraud_trend, x="TransactionDT", y="isFraud", title="Frauds Over Time")
# st.plotly_chart(fig_trend, use_container_width=True)

# # Risk score distribution
# if 'risk_score' in df.columns:
#     st.subheader("Risk Score Distribution")
#     fig_risk = px.histogram(df, x="risk_score", nbins=30)
#     st.plotly_chart(fig_risk, use_container_width=True)

# # Top risky cards
# if 'card1' in df.columns:
#     st.subheader("Top Risky Card1 IDs")
#     top_cards = df[df['isFraud'] == 1]['card1'].value_counts().nlargest(10).reset_index()
#     fig_card = px.bar(top_cards, x='index', y='card1', labels={'index': 'Card1', 'card1': 'Fraud Count'})
#     st.plotly_chart(fig_card, use_container_width=True)


import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="refresh")

# App title
st.title("Real-Time Fraud Detection Dashboard")

# Attempt to fetch data from API
API_URL = "https://your-api-url.com/transactions"
use_mock = False

try:
    response = requests.get(API_URL)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
except Exception as e:
    st.warning(f"Using mock data. API unavailable: {e}")
    use_mock = True

# If API fails, create mock data with spikes in fraud for visualization
use_mock = True  # Set to True for mock data

if use_mock:
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(100)][::-1]

    fraud_labels = []
    fraud_probabilities = []
    for i in range(100):
        if 20 <= i < 25:
            fraud_labels.append(3)  # Warning threshold (spike)
            fraud_probabilities.append(np.random.uniform(0.3, 0.5))  # Simulating fraud probability in the warning range
        elif 40 <= i < 43:
            fraud_labels.append(6)  # Critical threshold (higher spike)
            fraud_probabilities.append(np.random.uniform(0.6, 0.9))  # Simulating fraud probability in the critical range
        else:
            fraud_labels.append(np.random.binomial(n=1, p=0.05))  # Mostly non-fraud
            fraud_probabilities.append(np.random.uniform(0, 0.1))  # Low probability for non-fraud transactions

    # Create the mock DataFrame
    df = pd.DataFrame({
        'TransactionDT': timestamps,
        'TransactionAmt': np.random.uniform(10, 500, 100),
        'card1': np.random.choice([1111, 2222, 3333, 4444, 5555], 100),
        'isFraud': fraud_labels,
        'fraud_probability': fraud_probabilities,  # New column for fraud probability
        'risk_score': np.random.uniform(0, 1, 100)  # Random risk scores
    })

# Data/ Visualizations
# Sidebar filters
st.sidebar.header("Filter")
num_rows = st.sidebar.slider("Rows to show", 5, 100, 20)

# Summary statistics
st.subheader("Summary Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(df))
col2.metric("Fraudulent", int(df['isFraud'].sum()))
col3.metric("Average Amount", f"${df['TransactionAmt'].mean():.2f}")

# Show table
st.subheader("Live Transaction Table")
st.dataframe(df.tail(num_rows).sort_values("TransactionDT", ascending=False), use_container_width=True)

# Fraud trend with fraud probability vs time
st.subheader("Fraud Probability vs Time")

# Define thresholds
warning_threshold = 0.5  # Orange warning if fraud probability > 0.5
critical_threshold = 0.8  # Red alert if fraud probability > 0.8

# Create the base figure using fraud probability over time
fig_trend = px.line(
    df,
    x="TransactionDT",
    y="fraud_probability",
    title="Fraud Probability vs Time",
    labels={"fraud_probability": "Fraud Probability", "TransactionDT": "Time"},
)

# Add threshold lines for warning and critical thresholds
fig_trend.add_hline(
    y=warning_threshold,
    line_dash="dash",
    line_color="orange",
    annotation_text="Warning Threshold",
    annotation_position="bottom right"
)
fig_trend.add_hline(
    y=critical_threshold,
    line_dash="dash",
    line_color="red",
    annotation_text="Critical Threshold",
    annotation_position="top right"
)

# Improve layout
fig_trend.update_layout(
    yaxis_title="Fraud Probability",
    xaxis_title="Time",
    template="plotly_white"
)

# Display the plot
st.plotly_chart(fig_trend, use_container_width=True)

# Risk score distribution
if 'risk_score' in df.columns:
    st.subheader("Risk Score Distribution")
    fig_risk = px.histogram(df, x="risk_score", nbins=30)
    st.plotly_chart(fig_risk, use_container_width=True)

# Top risky cards
if 'card1' in df.columns:
    st.subheader("Top Risky Card1 IDs")
    top_cards = df[df['isFraud'] == 1]['card1'].value_counts().nlargest(10).reset_index()
    top_cards.columns = ['Card1', 'Fraud Count']  # Rename columns for clarity
    fig_card = px.bar(top_cards, x='Card1', y='Fraud Count', labels={'Card1': 'Card1', 'Fraud Count': 'Fraud Count'})
    st.plotly_chart(fig_card, use_container_width=True)

# # Status Breakdown 
# st.subheader("Transaction Status Breakdown")

# # Assume 'isFraud': 0 = valid, 1 = fraud, other = unassigned (for mock)
# df['status'] = df['isFraud'].apply(lambda x: 'Fraud' if x == 1 else ('Valid' if x == 0 else 'Unassigned'))

# status_counts = df['status'].value_counts().reset_index()
# status_counts.columns = ['Status', 'Count']

# fig_status = px.bar(status_counts, x='Status', y='Count', color='Status', title='Transaction Status Summary')
# st.plotly_chart(fig_status, use_container_width=True)

# # Unusual Trend Alerts 
# st.subheader("Unusual Trend Alerts (High Fraud Rate by Card1)")

# # Calculate fraud rate per card
# card_stats = df.groupby('card1').agg(
#     total=('isFraud', 'count'),
#     frauds=('isFraud', 'sum')
# ).reset_index()
# card_stats['fraud_rate'] = card_stats['frauds'] / card_stats['total']

# # Define threshold for unusual behavior (e.g., fraud rate > 0.5 and at least 3 frauds)
# alerts = card_stats[(card_stats['fraud_rate'] > 0.5) & (card_stats['frauds'] >= 3)]

# if not alerts.empty:
#     st.dataframe(alerts.sort_values('fraud_rate', ascending=False), use_container_width=True)
# else:
#     st.info("No unusual trends detected at this time.")

# # Ongoing Investigations Table 
# st.subheader("Ongoing Investigations")

# # For mock purposes, flag latest 5 frauds as 'under investigation'
# investigations = df[df['isFraud'] == 1].sort_values("TransactionDT", ascending=False).head(5)
# investigations = investigations[['TransactionDT', 'card1', 'TransactionAmt', 'risk_score']]
# investigations['Status'] = 'Under Investigation'

# st.dataframe(investigations.reset_index(drop=True), use_container_width=True)



# HOW TO RUN: streamlit run dashboard/dashboard.py