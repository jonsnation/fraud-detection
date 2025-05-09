import sys
import joblib
import shap
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

# === Path setup ===
project_root = Path(__file__).resolve().parents[1]
base_model_dir = project_root / "models" / "trained_model_manual_fields"
processed_dir = project_root / "data" / "processed"

# === Manual-entry field names (after encoding) ===
manual_feature_columns = [
    "TransactionAmt", "card1", "addr1", "dist1",
    "ProductCD_freq", "P_emaildomain_freq"
]

# === Load trained models ===
models = {
    "randomforest": joblib.load(base_model_dir / "randomforest_manual_model.pkl"),
    "xgboost": joblib.load(base_model_dir / "xgboost_manual_model.pkl"),
    "logisticregression": joblib.load(base_model_dir / "logisticregression_manual_model.pkl"),
    "gradientboosting": joblib.load(base_model_dir / "gradientboosting_manual_model.pkl"),
    "mlp": joblib.load(base_model_dir / "mlp_manual_model.pkl")
}

# === Load preprocessor and frequency maps ===
preprocessor = joblib.load(base_model_dir / "preprocessor_manual.pkl")
frequency_maps = joblib.load(processed_dir / "frequency_maps.pkl")
product_freq_map = frequency_maps.get("ProductCD", {})
email_freq_map = frequency_maps.get("P_emaildomain", {})

# === SHAP explainers for supported models ===
explainers = {}
for model_name in ["xgboost", "randomforest"]:
    try:
        explainers[model_name] = shap.TreeExplainer(models[model_name])
    except Exception as e:
        print(f"SHAP setup failed for {model_name}: {e}")

# === Manual prediction endpoint (Tab 1) ===
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    data = request.get_json(force=True)
    model_name = data.get("model", "xgboost").lower()

    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' is not available."}), 400

    model = models[model_name]

    # Raw user inputs
    raw_input = {k: v for k, v in data.items() if k != "model"}

    # Map categorical values to frequency-encoded values
    mapped_input = {
        "TransactionAmt": raw_input.get("TransactionAmt", 0),
        "card1": raw_input.get("card1", 0),
        "addr1": raw_input.get("addr1", 0),
        "dist1": raw_input.get("dist1", 0),
        "ProductCD_freq": product_freq_map.get(raw_input.get("ProductCD", ""), 0),
        "P_emaildomain_freq": email_freq_map.get(raw_input.get("P_emaildomain", "").lower(), 0)
    }

    input_df = pd.DataFrame([mapped_input])

    try:
        input_processed = preprocessor.transform(input_df)
        prediction = int(model.predict(input_processed)[0])
        probability = float(model.predict_proba(input_processed)[0][1])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # SHAP explanation (if available)
    shap_values = {}
    raw_shap_values = {}
    if model_name in explainers:
        try:
            shap_vals = explainers[model_name](input_processed)
            shap_array = shap_vals.values[0]
            shap_dict = {f: float(np.atleast_1d(v)[0]) for f, v in zip(manual_feature_columns, shap_array)}
            raw_shap_values = {k: round(v, 6) for k, v in shap_dict.items()}
            top_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
            shap_values = {k: round(v, 6) for k, v in top_features}
        except Exception as e:
            shap_values = {"warning": f"SHAP explanation failed: {str(e)}"}

    return jsonify({
        "model_used": model_name,
        "input": raw_input,
        "filled_input": input_df.iloc[0].to_dict(),
        "prediction": prediction,
        "fraud_probability": round(probability, 4),
        "shap_top_contributors": shap_values,
        "raw_shap_values": raw_shap_values
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
