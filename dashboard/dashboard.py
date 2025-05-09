import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import random
import json
from pathlib import Path
import joblib
import streamlit.components.v1 as components
import plotly.graph_objects as go

# Setup 
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    /* Card-like containers for expander */
    .streamlit-expanderHeader {
        font-weight: bold;
        background-color: #f5f5f5;
        padding: 6px 12px;
        border-radius: 4px;
    }

    /* Scrollable dataframe container */
    .element-container:has(.dataframe) {
        max-height: 350px;
        overflow-y: auto;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 5px;
        margin-bottom: 10px;
    }

    /* Expander inside scroll section */
    .stExpander > div[role="button"] {
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 4px;
        margin-top: 4px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        padding: 10px 20px;
    }

    /* Reduce padding in columns */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* Scrollable SHAP grid */
    .shap-grid-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Fraud Detection Dashboard")

API_URL = "http://localhost:5000/predict_full"
all_models = ["xgboost", "randomforest", "logisticregression", "gradientboosting", "mlp"]

if "history" not in st.session_state:
    st.session_state["history"] = []

if "stream_df" not in st.session_state:
    st.session_state["stream_df"] = pd.DataFrame()

tab1, tab2 = st.tabs(["Manual Entry", "Simulated Stream"])



# TAB 1 
with tab1:
    st.subheader("Enter Transaction Manually")

    # Load frequency maps from file (cached)
    @st.cache_data
    def load_frequency_maps():
        freq_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "frequency_maps.pkl"
        return joblib.load(freq_path)

    freq_maps = load_frequency_maps()
    product_map = freq_maps.get("ProductCD", {})
    email_map = freq_maps.get("P_emaildomain", {})

    all_models = ["xgboost", "randomforest", "logisticregression", "gradientboosting", "mlp"]

    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
            addr1 = st.number_input("Billing ZIP Code (addr1)", min_value=0, value=100)

        with col2:
            card1 = st.number_input("Card1 (Customer ID)", min_value=1000, value=3333)
            dist1 = st.number_input("Distance to Purchaser (dist1)", min_value=0.0, value=10.0)

        with col3:
            product = st.selectbox("Product Code", sorted(product_map.keys()))
            email = st.text_input("Email Domain", value="gmail.com")

        mode = st.selectbox("Choose Prediction Mode", ["Compare All Models"] + [m.upper() for m in all_models])
        submitted = st.form_submit_button("Predict")

    if submitted:
        transaction_input = {
            "TransactionAmt": amount,
            "card1": card1,
            "addr1": addr1,
            "dist1": dist1,
            "ProductCD": product,
            "P_emaildomain": email,
            "ProductCD_freq": product_map.get(product, 0.0),
            "P_emaildomain_freq": email_map.get(email.lower(), 0.0)
        }

        if mode == "Compare All Models":
            results = []
            with st.spinner("Sending data to all models..."):
                for model in all_models:
                    try:
                        response = requests.post("http://localhost:5000/predict_manual", json={**transaction_input, "model": model})
                        response.raise_for_status()
                        result = response.json()
                        results.append({
                            "Model": model.upper(),
                            "Prediction": "FRAUD" if result["prediction"] else "LEGITIMATE",
                            "Probability": result["fraud_probability"],
                        })
                    except Exception as e:
                        results.append({
                            "Model": model.upper(),
                            "Prediction": "ERROR",
                            "Probability": None,
                        })

            result_df = pd.DataFrame(results)
            st.subheader("Prediction Comparison")
            st.dataframe(result_df, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            fraud_votes = sum(1 for r in results if r["Prediction"] == "FRAUD")
            ensemble_label = "LIKELY FRAUD" if fraud_votes >= 3 else "LIKELY LEGITIMATE"
            confidence = fraud_votes / len(results)
            col1.metric("Total Models", len(results))
            col2.metric("Fraud Votes", fraud_votes)

            with col3:
                st.subheader("Ensemble Verdict")
                st.markdown(f"<h3 style='margin-bottom:0'>{ensemble_label}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:gray; font-size: 14px;'>Confidence: {confidence * 100:.1f}% agreement</p>", unsafe_allow_html=True)

            prob_chart = px.bar(
                result_df[result_df["Probability"].notnull()],
                x="Model", y="Probability", color="Prediction",
                title="Fraud Probability by Model",
                labels={"Probability": "Fraud Probability"},
                text_auto=".2f"
            )
            st.plotly_chart(prob_chart, use_container_width=True)

            st.session_state["history"].append({
                "input": transaction_input,
                "results": result_df.to_dict("records"),
                "ensemble": ensemble_label,
                "confidence": confidence
            })

        else:
            model_name = mode.lower()
            with st.spinner(f"Predicting with {mode}..."):
                try:
                    response = requests.post("http://localhost:5000/predict_manual", json={**transaction_input, "model": model_name})
                    response.raise_for_status()
                    result = response.json()

                    prediction = "FRAUD" if result["prediction"] else "LEGITIMATE"
                    fraud_probability = result["fraud_probability"]

                    st.subheader(f"Prediction using {mode}")
                    st.write(f"Prediction: **{prediction}**")
                    st.metric("Fraud Probability", f"{fraud_probability*100:.2f}%")

                    st.session_state["history"].append({
                        "input": transaction_input,
                        "results": [{
                            "Model": model_name.upper(),
                            "Prediction": prediction,
                            "Probability": fraud_probability
                        }],
                        "ensemble": prediction,
                        "confidence": 1.0
                    })

                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# TAB 2
with tab2:
    import pandas as pd
    import requests
    import plotly.graph_objects as go
    import streamlit as st

    st.subheader("Real Transaction Stream from Training Data")

    stream_api_url = "http://localhost:5000/predict_stream"
    gnn_context_api_url = "http://localhost:5000/predict_gnn_with_context"
    stream_source_url = "http://localhost:5000/get_stream"

    tree_models = ["xgboost", "randomforest", "logisticregression", "gradientboosting", "mlp"]

    stream_features = [
        "TransactionAmt", "TransactionDT", "card1", "card4_freq", "card6_freq",
        "addr1", "dist1", "P_emaildomain_freq", "R_emaildomain_freq", "M1_freq",
        "M4_freq", "M5_freq", "M6_freq", "M9_freq", "C1", "C2", "C8", "C11", "V18",
        "V21", "V97", "V133", "V189", "V200", "V258", "V282", "V294", "V312",
        "DeviceType_freq", "id_15_freq", "id_28_freq", "id_29_freq", "id_31_freq",
        "id_35_freq", "id_36_freq", "id_37_freq", "id_38_freq"
    ]

    gnn_features = [
        'V233', 'V132', 'C5', 'V261', 'V134', 'V302', 'DeviceInfo', 'V184', 'V183', 'V239', 'V224', 'V291',
        'id_31', 'V228', 'V319', 'V185', 'id_16', 'V235', 'V258', 'id_01', 'V213', 'V9', 'V98', 'V320', 'V232',
        'id_02', 'V204', 'V115', 'V172', 'V252', 'V276', 'V210', 'V180', 'id_36', 'V46', 'V51', 'V103',
        'TransactionDT', 'TransactionAmt', 'ProductCD', 'card4', 'C8', 'C9', 'card3', 'C6'
    ]

    if st.button("Load Latest Stream Batch"):
        try:
            stream = requests.get(stream_source_url).json()
            df_all = pd.DataFrame(stream)
            st.session_state["stream_tree"] = df_all[df_all.get("type") == "tree"].copy()
            st.session_state["stream_gnn"] = df_all[df_all.get("type") == "gnn"].copy()
        except Exception as e:
            st.error(f"Failed to fetch stream: {e}")

    df_tree = st.session_state.get("stream_tree", pd.DataFrame())
    df_gnn = st.session_state.get("stream_gnn", pd.DataFrame())

    # === Tree-Based Section ===
    if not df_tree.empty:
        st.markdown("### Tree-Based Feature View + Prediction")
        tree_cols = [c for c in stream_features if c in df_tree.columns]
        st.dataframe(df_tree[tree_cols], use_container_width=True, height=300)

        selected_tree_row = st.selectbox("Select transaction for Tree Models", df_tree.index, key="select_tree")

        if st.button("Predict with Tree Models"):
            row = df_tree.loc[selected_tree_row].to_dict()
            clean_row = {k: row.get(k, 0) for k in stream_features}
            results = []

            for model in tree_models:
                try:
                    payload = {**clean_row, "model": model}
                    response = requests.post(stream_api_url, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    results.append({
                        "Model": model.upper(),
                        "Prediction": "FRAUD" if result["prediction"] else "LEGITIMATE",
                        "Fraud Probability": f"{result['fraud_probability'] * 100:.2f}%",
                        "AE Flag": "YES" if result.get("autoencoder_flagged") else "NO",
                        "Reconstruction Error": f"{result.get('reconstruction_error', 0):.6f}"
                    })
                except Exception as e:
                    st.error(f"{model.upper()} failed: {e}")
                    results.append({
                        "Model": model.upper(),
                        "Prediction": "ERROR",
                        "Fraud Probability": "N/A",
                        "AE Flag": "N/A",
                        "Reconstruction Error": "N/A"
                    })

            st.markdown("#### Tree Model Predictions")
            st.dataframe(pd.DataFrame(results), use_container_width=True)

    # === GNN Section ===
    if not df_gnn.empty:
        st.markdown("### GNN Feature View + Prediction")
        gnn_cols = [c for c in gnn_features if c in df_gnn.columns]
        st.dataframe(df_gnn[gnn_cols], use_container_width=True, height=300)

        selected_gnn_row = st.selectbox("Select transaction for GNN", df_gnn.index, key="select_gnn")

        if st.button("Predict with GNN (Full Context)"):
            row = df_gnn.loc[selected_gnn_row].to_dict()
            input_gnn = {k: row.get(k, 0) for k in gnn_features}
            try:
                res = requests.post(gnn_context_api_url, json=input_gnn)
                res.raise_for_status()
                r = res.json()
                label = "FRAUD" if r["prediction"] else "LEGITIMATE"
                color = "red" if r["prediction"] else "green"

                st.markdown(f"<h3 style='text-align:center; color:{color}'>{label}</h3>", unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=r["fraud_probability"] * 100,
                    title={'text': "Fraud Probability", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "salmon"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"GNN prediction failed: {e}")

    # === Fallback Message ===
    if df_tree.empty and df_gnn.empty:
        st.warning("No data loaded. Click 'Load Latest Stream Batch' to snapshot live stream.")



# HOW TO RUN: streamlit run dashboard/dashboard.py