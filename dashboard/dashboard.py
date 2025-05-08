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

tab1, tab2, tab3 = st.tabs(["Manual Entry", "Simulated Stream", "Advanced Test Case"])



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
        # Apply frequency encoding using real maps
        transaction_input = {
            "TransactionAmt": amount,
            "card1": card1,
            "addr1": addr1,
            "dist1": dist1,
            "ProductCD_freq": product_map.get(product, 0.0),
            "P_emaildomain_freq": email_map.get(email.lower(), 0.0)
        }

        if mode == "Compare All Models":
            results = []
            shap_contributors = {}
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
                        shap_contributors[model] = result.get("shap_top_contributors", {})
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

            st.subheader("SHAP Top Contributors (per Model)")
            shap_cols = st.columns(2)
            for i, model in enumerate(all_models):
                if shap_contributors.get(model):
                    with shap_cols[i % 2]:
                        shap_df = pd.DataFrame.from_dict(shap_contributors[model], orient='index', columns=["SHAP Value"])
                        shap_df["Feature"] = shap_df.index
                        shap_df = shap_df.sort_values("SHAP Value", key=abs, ascending=False)
                        st.markdown(f"Model: {model.upper()}")
                        shap_fig = px.bar(shap_df, x="Feature", y="SHAP Value", title=f"{model.upper()} SHAP")
                        st.plotly_chart(shap_fig, use_container_width=True)

            st.subheader("SHAP Feature Impact Heatmap")
            combined_shap = []
            for model, shap_data in shap_contributors.items():
                for feature, value in shap_data.items():
                    combined_shap.append({"Model": model.upper(), "Feature": feature, "SHAP Value": value})
            if combined_shap:
                heat_df = pd.DataFrame(combined_shap)
                heat_pivot = heat_df.pivot(index="Feature", columns="Model", values="SHAP Value").fillna(0)
                fig_heatmap = px.imshow(
                    heat_pivot,
                    title="SHAP Feature Impact Heatmap",
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

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

                    shap_dict = result.get("shap_top_contributors", {})
                    if shap_dict:
                        shap_df = pd.DataFrame.from_dict(shap_dict, orient="index", columns=["SHAP Value"])
                        shap_df["Feature"] = shap_df.index
                        shap_df = shap_df.sort_values("SHAP Value", key=abs, ascending=False)
                        st.subheader(f"{mode} SHAP Explanation")
                        shap_fig = px.bar(shap_df, x="Feature", y="SHAP Value")
                        st.plotly_chart(shap_fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    st.subheader("Prediction History")
    with st.container():
        for idx, item in enumerate(reversed(st.session_state["history"][-5:])):
            with st.expander(f"Transaction #{len(st.session_state['history']) - idx}"):
                st.json(item["input"])
                hist_df = pd.DataFrame(item["results"])
                st.dataframe(hist_df, use_container_width=True)
                st.caption(f"Ensemble: {item['ensemble']} — {item['confidence']*100:.1f}% agreement")

# TAB 2
with tab2:
    import random
    import pandas as pd
    import requests
    import json
    import plotly.graph_objects as go
    from pathlib import Path

    st.subheader("Real Transaction Stream from Training Data")

    # API Endpoints
    stream_api_url = "http://localhost:5000/predict_stream"
    gnn_context_api_url = "http://localhost:5000/predict_gnn_with_context"

    # OS-independent file paths
    project_root = Path(__file__).resolve().parent.parent
    gnn_dir = project_root / "gnn"
    reduced_features_path = gnn_dir / "reduced_features.csv"
    balanced_labels_path = gnn_dir / "balanced_labels.csv"
    tree_real_data_path = project_root / "data" / "processed" / "X_subset_features.csv"

    tree_models = ["xgboost", "randomforest", "logisticregression", "gradientboosting", "mlp"]
    gnn_model = "fraudgnn"

    stream_features = [
        "TransactionAmt", "TransactionDT", "card1", "card4_freq", "card6_freq",
        "addr1", "dist1", "P_emaildomain_freq", "R_emaildomain_freq",
        "M1_freq", "M4_freq", "M5_freq", "M6_freq", "M9_freq",
        "C1", "C2", "C8", "C11", "V18", "V21", "V97", "V133", "V189", "V200",
        "V258", "V282", "V294", "V312", "DeviceType_freq", "id_15_freq",
        "id_28_freq", "id_29_freq", "id_31_freq", "id_35_freq", "id_36_freq",
        "id_37_freq", "id_38_freq"
    ]

    gnn_features = [
        'V233', 'V132', 'C5', 'V261', 'V134', 'V302', 'DeviceInfo', 'V184', 'V183', 'V239', 'V224', 'V291',
        'id_31', 'V228', 'V319', 'V185', 'id_16', 'V235', 'V258', 'id_01', 'V213', 'V9', 'V98', 'V320', 'V232',
        'id_02', 'V204', 'V115', 'V172', 'V252', 'V276', 'V210', 'V180', 'id_36', 'V46', 'V51', 'V103',
        'TransactionDT', 'TransactionAmt', 'ProductCD', 'card4', 'C8', 'C9', 'card3', 'C6'
    ]

    @st.cache_data
    def load_real_tree_stream(n=10):
        df = pd.read_csv(tree_real_data_path)
        return df.sample(n=n, random_state=42).reset_index(drop=True)

    @st.cache_data
    def load_gnn_samples(n=10):
        X = pd.read_csv(reduced_features_path)
        y = pd.read_csv(balanced_labels_path).squeeze()
        df = X.copy()
        df["isFraud"] = y
        return df.sample(n=n, random_state=42).reset_index(drop=True)

    # Top Buttons: Tree + GNN Load
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Real Transaction Batch", key="load_real_stream_btn"):
            st.session_state["stream_df"] = load_real_tree_stream(10)

    with col2:
        if st.button("Load 10 GNN Samples", key="load_gnn_samples"):
            st.session_state["gnn_df"] = load_gnn_samples(10)

    # Tree Model Prediction Section
    if not st.session_state.get("stream_df", pd.DataFrame()).empty:
        df = st.session_state["stream_df"]
        st.markdown("### Tree-Based Feature View + Prediction")
        tree_cols = [c for c in stream_features if c in df.columns]
        st.dataframe(df[tree_cols], use_container_width=True, height=300)

        selected_tree_row = st.selectbox("Select transaction for Tree Models", df.index, key="select_tree")

        if st.button("Predict with Tree Models", key="predict_tree_btn"):
            row = df.loc[selected_tree_row].to_dict()
            tree_results = []
            with st.spinner("Predicting with Autoencoder-enhanced Models..."):
                for model in tree_models:
                    try:
                        payload = {**row, "model": model}
                        response = requests.post(stream_api_url, json=payload)
                        response.raise_for_status()
                        result = response.json()
                        pred = "FRAUD" if result["prediction"] else "LEGITIMATE"
                        ae_flag = "YES" if result["autoencoder_flagged"] else "NO"
                        tree_results.append({
                            "Model": model.upper(),
                            "Prediction": pred,
                            "Fraud Probability": f"{result['fraud_probability']*100:.2f}%",
                            "AE Flag": ae_flag,
                            "Reconstruction Error": f"{result['reconstruction_error']:.6f}"
                        })
                    except Exception as e:
                        tree_results.append({
                            "Model": model.upper(),
                            "Prediction": "ERROR",
                            "Fraud Probability": "N/A",
                            "AE Flag": "N/A",
                            "Reconstruction Error": "N/A"
                        })

            st.markdown("#### Autoencoder-Augmented Tree Model Results")
            st.dataframe(pd.DataFrame(tree_results), use_container_width=True)

    # GNN Model Prediction Section
    gnn_df = st.session_state.get("gnn_df", pd.DataFrame())
    if not gnn_df.empty:
        st.markdown("### GNN Feature View + Prediction")
        gnn_cols = [c for c in gnn_features if c in gnn_df.columns]
        st.dataframe(gnn_df[gnn_cols], use_container_width=True, height=300)

        selected_gnn_row = st.selectbox("Select transaction for GNN", gnn_df.index, key="select_gnn")

        if st.button("Predict with GNN (Full Context)", key="predict_gnn_context_btn"):
            row = gnn_df.loc[selected_gnn_row].to_dict()
            input_gnn = {k: row.get(k, 0) for k in gnn_features}
            try:
                response = requests.post(gnn_context_api_url, json=input_gnn)
                result = response.json()

                label = "FRAUD" if result["prediction"] else "LEGITIMATE"
                color = "red" if result["prediction"] else "green"

                st.markdown(f"<h3 style='text-align:center; color:{color}'>{label}</h3>", unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["fraud_probability"] * 100,
                    title={'text': "Fraud Probability", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "crimson" if result["prediction"] else "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "salmon"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

                actual = gnn_df.loc[selected_gnn_row, "isFraud"]
                verdict = "Correct" if result["prediction"] == actual else "Incorrect"
                st.markdown(f"**Prediction vs Ground Truth:** `{verdict}`")

            except Exception as e:
                st.error(f"GNN Context Prediction Failed: {str(e)}")


# TAB 3
with tab3:
    st.subheader("Advanced Test Case (Full Feature Input)")

    mode = st.radio("Select Input Mode", ["Use Sample Fraud-like Input", "Paste Custom JSON"])

    sample_input = None

    if mode == "Use Sample Fraud-like Input":
        import random
        sample_input = {
            "TransactionAmt": round(random.uniform(10000, 90000), 2),
            "card1": random.randint(700000, 999999),
            "addr1": random.randint(250, 350),
            "dist1": round(random.uniform(100, 300), 2),
            "model": "xgboost",
            **{f"C{i}": random.randint(0, 5) for i in range(1, 15)},
            **{f"D{i}": random.randint(10, 100) for i in range(1, 16)},
            **{f"V{i}": round(random.uniform(0, 1), 4) for i in range(1, 340)}
        }

        st.code(json.dumps(sample_input, indent=2), language="json")

    else:
        user_input = st.text_area("Paste JSON input here:", height=300)

        if user_input:
            try:
                cleaned_input = (
                    user_input.strip()
                    .replace("“", '"').replace("”", '"')
                    .replace("‘", "'").replace("’", "'")
                    .replace("\xa0", " ")
                )
                sample_input = json.loads(cleaned_input)
                st.success("JSON parsed successfully.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                sample_input = None

    if sample_input and st.button("Run Prediction"):
        with st.spinner("Sending input to API..."):
            try:
                response = requests.post(API_URL, json=sample_input)
                response.raise_for_status()
                result = response.json()

                prediction = "FRAUD" if result["prediction"] else "LEGITIMATE"
                fraud_prob = result["fraud_probability"] * 100

                st.success(f"Prediction: {prediction}")
                st.metric("Fraud Probability", f"{fraud_prob:.2f}%")

                st.subheader("Top SHAP Contributors")
                shap_dict = result.get("shap_top_contributors", {})
                if shap_dict:
                    shap_df = pd.DataFrame.from_dict(shap_dict, orient='index', columns=["SHAP Value"])
                    shap_df["Feature"] = shap_df.index
                    shap_df = shap_df.sort_values("SHAP Value", key=abs, ascending=False)
                    fig = px.bar(shap_df, x="Feature", y="SHAP Value", title="Top SHAP Feature Contributions")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No SHAP values returned.")

                with st.expander("Full Model Input Used (filled_input)"):
                    st.json(result.get("filled_input", {}))

            except Exception as e:
                st.error(f"API call failed: {e}")

# HOW TO RUN: streamlit run dashboard/dashboard.py