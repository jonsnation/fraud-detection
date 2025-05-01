import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import random
import json



# --- Setup ---
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

# ========== TAB 1 ==========
with tab1:
    st.subheader("Enter Transaction Manually")
    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, help="Total amount of the transaction")
            addr1 = st.number_input("Billing ZIP Code (addr1)", min_value=0, value=100, help="First part of the billing address ZIP code")

        with col2:
            card1 = st.number_input("Card1 (Customer ID)", min_value=1000, value=3333, help="Primary card/customer identifier")
            dist1 = st.number_input("Distance to Purchaser (dist1)", min_value=0.0, value=10.0, help="Distance between billing and purchasing locations")

        with col3:
            product = st.selectbox("Product Code", ["W", "C", "R", "H", "S"], help="Type of product purchased")
            email = st.text_input("Email Domain",value="gmail.com",help="Domain of the user's email address (e.g., gmail.com, yahoo.com).")


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
        }

        if mode == "Compare All Models":
            results = []
            shap_contributors = {}

            with st.spinner("Sending data to all models..."):
                for model in all_models:
                    payload = {**transaction_input, "model": model}
                    try:
                        response = requests.post(API_URL, json=payload)
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

            # Save to unified history
            st.session_state["history"].append({
                "input": transaction_input,
                "results": result_df.to_dict("records"),
                "ensemble": ensemble_label,
                "confidence": confidence
            })

            # SHAP and Heatmap
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

            st.subheader("SHAP Feature Heatmap")
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
            # Single model path
            model_name = mode.lower()
            with st.spinner(f"Predicting with {mode}..."):
                try:
                    response = requests.post(API_URL, json={**transaction_input, "model": model_name})
                    response.raise_for_status()
                    result = response.json()

                    prediction = "FRAUD" if result["prediction"] else "LEGITIMATE"
                    fraud_probability = result["fraud_probability"]

                    st.subheader(f"Prediction using {mode}")
                    st.write(f"Prediction: **{prediction}**")
                    st.metric("Fraud Probability", f"{fraud_probability*100:.2f}%")

                    # Save history for single-model case
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

    # Unified prediction history for all modes
    st.subheader("Prediction History")
    with st.container():
        for idx, item in enumerate(reversed(st.session_state["history"][-5:])):
            with st.expander(f"Transaction #{len(st.session_state['history']) - idx}"):
                st.json(item["input"])
                hist_df = pd.DataFrame(item["results"])
                st.dataframe(hist_df, use_container_width=True)
                st.caption(f"Ensemble: {item['ensemble']} — {item['confidence']*100:.1f}% agreement")


# ========== TAB 2 ==========
with tab2:
    st.subheader("Simulated Kafka Stream")

    def generate_fraud_like_transaction():
        return {
            "TransactionAmt": round(random.uniform(10000, 50000), 2),
            "card1": random.choice([9999, 8888, 7777]),
            "addr1": random.randint(200, 400),
            "dist1": round(random.uniform(200, 500), 2),
            "ProductCD": random.choice(["R", "S"]),
            "P_emaildomain": "anonymous.com",
            "RiskTag": "HIGH"
        }

    def simulate_kafka_stream():
        now = datetime.now()
        data = []

        # Inject 3 realistic high-risk samples
        for i in range(3):
            transaction = generate_fraud_like_transaction()
            transaction["TransactionDT"] = now - timedelta(seconds=5 * i)
            data.append(transaction)

        # Add 7 normal samples
        for i in range(3, 10):
            transaction = {
                "TransactionDT": now - timedelta(seconds=5 * i),
                "TransactionAmt": round(random.uniform(10, 300), 2),
                "card1": random.choice([1111, 2222, 3333, 4444, 5555]),
                "addr1": random.randint(1, 200),
                "dist1": round(random.uniform(0, 15), 2),
                "ProductCD": random.choice(["W", "C", "H"]),
                "P_emaildomain": random.choice(["gmail.com", "yahoo.com", "hotmail.com"]),
                "RiskTag": "LOW"
            }
            data.append(transaction)

        return pd.DataFrame(data)

    if st.button("Generate New Stream Batch"):
        st.session_state["stream_df"] = simulate_kafka_stream()

    if not st.session_state["stream_df"].empty:
        def highlight_risky(row):
            if row["RiskTag"] == "HIGH":
                return ['background-color: #aa4400'] * len(row)
            else:
                return [''] * len(row)

        styled_stream = st.session_state["stream_df"].style.apply(highlight_risky, axis=1)
        st.dataframe(styled_stream, height=350, use_container_width=True)

        with st.expander("Predict Fraud for a Streamed Transaction"):
            selected_row = st.selectbox("Select a transaction to simulate:", st.session_state["stream_df"].index)

            if st.button("Predict selected from stream"):
                selected_input = st.session_state["stream_df"].loc[selected_row].to_dict()
                selected_input.pop("TransactionDT")
                selected_input.pop("RiskTag")

                model_results = []
                fraud_model_names = []

                with st.spinner("Predicting with all models..."):
                    for model in all_models:
                        payload = {**selected_input, "model": model}
                        try:
                            response = requests.post(API_URL, json=payload)
                            response.raise_for_status()
                            result = response.json()
                            prediction = "FRAUD" if result["prediction"] == 1 else "LEGITIMATE"
                            if result["prediction"] == 1:
                                fraud_model_names.append(model.upper())
                            model_results.append({
                                "Model": model.upper(),
                                "Prediction": prediction,
                                "Fraud Probability": f"{result['fraud_probability']*100:.2f}%"
                            })
                        except Exception as e:
                            model_results.append({
                                "Model": model.upper(),
                                "Prediction": "ERROR",
                                "Fraud Probability": "N/A"
                            })

                model_df = pd.DataFrame(model_results)
                st.dataframe(model_df, use_container_width=True)

                # Ensemble decision
                fraud_votes = len(fraud_model_names)
                ensemble_label = "LIKELY FRAUD" if fraud_votes >= 3 else "LIKELY LEGITIMATE"
                confidence = fraud_votes / len(all_models)

                st.markdown(f"**Ensemble Verdict:** {ensemble_label}")
                st.markdown(f"**Models Voting FRAUD:** {', '.join(fraud_model_names) or 'None'}")
                st.metric("Fraud Vote Confidence", f"{confidence*100:.1f}% agreement")

                # Display SHAP Top Contributors if available
                st.subheader("SHAP Top Contributors per Model")

                shap_cols = st.columns(2)
                for i, model in enumerate(all_models):
                    matching_result = next((r for r in model_results if r["Model"] == model.upper()), None)
                    if matching_result and prediction != "ERROR":
                        try:
                            # Fetch SHAP from backend call again with explain=True (or reuse result)
                            payload = {**selected_input, "model": model}
                            response = requests.post(API_URL, json=payload)
                            response.raise_for_status()
                            result = response.json()
                            shap_dict = result.get("shap_top_contributors", {})
                            if shap_dict:
                                shap_df = pd.DataFrame.from_dict(shap_dict, orient='index', columns=["SHAP Value"])
                                shap_df["Feature"] = shap_df.index
                                shap_df = shap_df.sort_values("SHAP Value", key=abs, ascending=False)
                                with shap_cols[i % 2]:
                                    st.markdown(f"**{model.upper()} SHAP**")
                                    fig = px.bar(shap_df, x="Feature", y="SHAP Value", title=f"{model.upper()} Explanation")
                                    st.plotly_chart(fig, use_container_width=True)
                        except:
                            pass

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