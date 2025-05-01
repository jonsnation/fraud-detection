import sys
import joblib
import shap
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model  # Only used if you're loading Keras models like MLP

# We're setting up a Flask app so that we can send transactions via HTTP and get predictions back
app = Flask(__name__)

# These are the paths to where your models and feature metadata are stored
project_root = Path(__file__).resolve().parents[1]
base_model_dir = project_root / "models" / "trained_model"
processed_dir = project_root / "data" / "processed"

# Load all the machine learning models you’ve trained and saved previously
models = {
    "randomforest": joblib.load(base_model_dir / "randomforest" / "randomforest_full_model.pkl"),
    "xgboost": joblib.load(base_model_dir / "xgboost" / "xgboost_full_model.pkl"),
    "logisticregression": joblib.load(base_model_dir / "logisticregression" / "logisticregression_full_model.pkl"),
    "gradientboosting": joblib.load(base_model_dir / "gradientboosting" / "gradientboosting_full_model.pkl"),
    "mlp": joblib.load(base_model_dir / "mlp" / "mlp_full_model.pkl")
}

# Load the feature list and preprocessing pipeline so we can format new input data correctly
feature_columns = joblib.load(processed_dir / "feature_columns.pkl")
preprocessor = joblib.load(base_model_dir / "preprocessor.pkl")

# If the model supports it, we'll also set up SHAP for explainability
explainers = {}
for model_name in ["xgboost", "randomforest"]:
    try:
        explainers[model_name] = shap.TreeExplainer(models[model_name])
    except Exception as e:
        print(f"SHAP failed for {model_name}: {e}")

# This is the route you'll send requests to when you want a fraud prediction
@app.route('/predict_full', methods=['POST'])
def predict_full():
    # We expect a JSON object, and we'll pull out the model name (defaults to xgboost if not specified)
    data = request.get_json(force=True)
    model_name = data.get("model", "xgboost").lower()

    # If the model requested doesn’t exist, return an error
    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' is not available."}), 400

    model = models[model_name]

    # Remove the model name from the input and create a DataFrame from the remaining fields
    transaction_data = {k: v for k, v in data.items() if k != "model"}
    input_df = pd.DataFrame([transaction_data])

    # Ensure the DataFrame has all required features, and fill in any missing ones with 0
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    try:
        # Apply the same preprocessing we used during training
        input_processed = preprocessor.transform(input_df)

        # Make a prediction and calculate the probability that it's fraud
        prediction = int(model.predict(input_processed)[0])
        probability = float(model.predict_proba(input_processed)[0][1])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # If the model supports SHAP, we'll generate an explanation for why it made this prediction
    shap_values = {}
    raw_shap_values = {}
    if model_name in explainers:
        try:
            shap_vals = explainers[model_name](input_processed)
            shap_array = shap_vals.values[0]
            shap_dict = {f: float(np.atleast_1d(v)[0]) for f, v in zip(feature_columns, shap_array)}

            # Include the full SHAP breakdown and the top 10 most influential features
            raw_shap_values = {k: round(v, 6) for k, v in shap_dict.items()}
            top_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            shap_values = {k: round(v, 6) for k, v in top_features}
        except Exception as e:
            shap_values = {"warning": f"SHAP explanation failed: {str(e)}"}

    # Now we return the prediction along with some extra context to help understand the result
    return jsonify({
        "model_used": model_name,
        "input": transaction_data,
        "filled_input": input_df.iloc[0].to_dict(),
        "prediction": prediction,
        "fraud_probability": round(probability, 4),
        "shap_top_contributors": shap_values,
        "raw_shap_values": raw_shap_values
    })

# This just starts the Flask development server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
