import sys
import joblib
import shap
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
import pickle
from itertools import combinations, islice
from tensorflow.keras.models import load_model
import traceback
import warnings

warnings.filterwarnings("ignore")


# === Resolve project root ===
project_root = Path(__file__).resolve()
while not (project_root / "gnn").exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.append(str(project_root))

# === Init Flask app ===
app = Flask(__name__)

# === Define GNN model class ===
class FraudGNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.classifier = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return torch.sigmoid(self.classifier(x))

# === Load Autoencoder and Assets ===
autoencoder = load_model(project_root / "models" / "trained_model" / "autoencoder_selected_model.keras")
ae_scaler = joblib.load(project_root / "models" / "trained_model" / "autoencoder_scaler.pkl")
ae_threshold = joblib.load(project_root / "models" / "trained_model" / "autoencoder_threshold.pkl")

# === Define correct feature order for autoencoder ===
autoencoder_features = [
    "TransactionAmt", "TransactionDT", "card1", "card4_freq", "card6_freq",
    "addr1", "dist1", "P_emaildomain_freq", "R_emaildomain_freq",
    "M1_freq", "M4_freq", "M5_freq", "M6_freq", "M9_freq",
    "C1", "C2", "C8", "C11",
    "V18", "V21", "V97", "V133", "V189", "V200", "V258", "V282", "V294", "V312",
    "DeviceType_freq", "id_15_freq", "id_28_freq", "id_29_freq",
    "id_31_freq", "id_35_freq", "id_36_freq", "id_37_freq", "id_38_freq"
]

# Load exact column names as saved during training
ae_feature_names = joblib.load(project_root / "models" / "trained_model" / "autoencoder_feature_names.pkl")

def flag_anomaly_with_autoencoder(row):
    df = pd.DataFrame([row])
    
    # Add missing features with 0.0, ensure correct order
    for col in ae_feature_names:
        if col not in df:
            df[col] = 0.0
    df_clean = df[ae_feature_names]  # Reorder strictly

    # Transform safely
    X_scaled = ae_scaler.transform(df_clean)
    X_reconstructed = autoencoder.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(X_scaled - X_reconstructed), axis=1)[0]
    is_anomaly = mse > ae_threshold
    return float(mse), bool(is_anomaly)


# === Load GNN Model ===
gnn_model = pickle.load(open(project_root / "gnn" / "fraudgnn_model.pkl", "rb"))
gnn_model.load_state_dict(torch.load(project_root / "gnn" / "best_fraudgnn_model.pt", map_location="cpu"))
gnn_model.eval()

# === Feature Definitions ===
manual_model_dir = project_root / "models" / "trained_model_manual_fields"
stream_model_dir = project_root / "models" / "trained_model_selected_subset"
processed_dir = project_root / "data" / "processed"

manual_features = ["TransactionAmt", "card1", "addr1", "dist1", "ProductCD_freq", "P_emaildomain_freq"]
stream_features = autoencoder_features

gnn_features = [
    'V233', 'V132', 'C5', 'V261', 'V134', 'V302', 'DeviceInfo', 'V184', 'V183', 'V239', 'V224', 'V291',
    'id_31', 'V228', 'V319', 'V185', 'id_16', 'V235', 'V258', 'id_01', 'V213', 'V9', 'V98', 'V320', 'V232',
    'id_02', 'V204', 'V115', 'V172', 'V252', 'V276', 'V210', 'V180', 'id_36', 'V46', 'V51', 'V103',
    'TransactionDT', 'TransactionAmt', 'ProductCD', 'card4', 'C8', 'C9', 'card3', 'C6'
]
gnn_edge_fields = ['card1', 'addr1', 'addr2', 'P_emaildomain', 'DeviceType', 'id_17', 'id_28']

# === Load Tree-Based Models and Explainers ===
manual_preprocessor = joblib.load(manual_model_dir / "preprocessor_manual.pkl")
stream_preprocessor = joblib.load(stream_model_dir / "preprocessor.pkl")
freq_maps = joblib.load(processed_dir / "frequency_maps.pkl")

product_freq = freq_maps.get("ProductCD", {})
email_freq = freq_maps.get("P_emaildomain", {})

manual_models = {n: joblib.load(manual_model_dir / f"{n}_manual_model.pkl") for n in ["randomforest", "xgboost", "logisticregression", "gradientboosting", "mlp"]}
stream_models = {n: joblib.load(stream_model_dir / f"{n}_selected_model.pkl") for n in ["randomforest", "xgboost", "logisticregression", "gradientboosting", "mlp"]}

manual_explainers, stream_explainers = {}, {}
for name in ["xgboost", "randomforest"]:
    try:
        manual_explainers[name] = shap.TreeExplainer(manual_models[name])
        stream_explainers[name] = shap.TreeExplainer(stream_models[name])
    except Exception:
        pass

def create_edges(feature_column, df, max_edges_per_group=100):
    edge_list = []
    groups = df.groupby(feature_column).indices
    for _, indices in groups.items():
        if len(indices) < 2:
            continue
        pair_generator = combinations(indices, 2)
        limited_pairs = list(islice(pair_generator, max_edges_per_group))
        edge_list.extend(limited_pairs)
    return torch.tensor(edge_list, dtype=torch.int32).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.int32)

# === Tab 1 Manual Prediction Endpoint ===

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    data = request.get_json(force=True)
    model_name = data.get("model", "xgboost").lower()
    model = manual_models.get(model_name)
    if not model:
        return jsonify({"error": f"Model '{model_name}' not found"}), 400

    # Feature engineering
    row = {
        "TransactionAmt": data.get("TransactionAmt", 0),
        "card1": data.get("card1", 0),
        "addr1": data.get("addr1", 0),
        "dist1": data.get("dist1", 0),
        "ProductCD_freq": product_freq.get(data.get("ProductCD", ""), 0),
        "P_emaildomain_freq": email_freq.get(data.get("P_emaildomain", "").lower(), 0),
    }

    df = pd.DataFrame([row]).reindex(columns=manual_features, fill_value=0)
    X = manual_preprocessor.transform(df)

    try:
        y_pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "model_used": model_name,
        "input": data,
        "filled_input": df.iloc[0].to_dict(),
        "prediction": y_pred,
        "fraud_probability": round(prob, 4)
    })


# === Endpoint: /predict_stream ===
@app.route('/predict_stream', methods=['POST'])
def predict_stream():
    print("[HIT] /predict_stream endpoint")
    try:
        data = request.get_json(force=True)
        model_name = data.get("model", "xgboost").lower()
        print(f"\n[REQUEST] Model: {model_name}")

        if model_name not in stream_models:
            print(f"[ERROR] Model '{model_name}' not found")
            return jsonify({"error": f"Model '{model_name}' not found"}), 400

        model = stream_models[model_name]

        df = pd.DataFrame([data]).reindex(columns=stream_features, fill_value=0)
        print("[DEBUG] DataFrame for prediction:\n", df)

        X = stream_preprocessor.transform(df)
        print("[DEBUG] Transformed shape:", X.shape)

        y_pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

        mse, flagged = flag_anomaly_with_autoencoder(data)

        return jsonify({
            "model_used": model_name,
            "input": data,
            "filled_input": df.iloc[0].to_dict(),
            "prediction": y_pred,
            "fraud_probability": round(prob, 4),
            "autoencoder_flagged": flagged,
            "reconstruction_error": round(mse, 6)
        })

    except Exception as e:
        import traceback
        print("\n[EXCEPTION]")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# === Endpoint: /predict_gnn_with_context ===
@app.route('/predict_gnn_with_context', methods=['POST'])
def predict_gnn_with_context():
    try:
        X = pd.read_csv(project_root / "gnn" / "reduced_features.csv").tail(100000).reset_index(drop=True)
        y = pd.read_csv(project_root / "gnn" / "balanced_labels.csv").squeeze().tail(len(X)).reset_index(drop=True)

        new_txn = request.get_json(force=True)
        if not isinstance(new_txn, dict):
            return jsonify({"error": "Invalid input. Expecting JSON object"}), 400

        df_new = pd.DataFrame([new_txn])
        df_all = pd.concat([X, df_new], ignore_index=True)

        for col in gnn_features:
            if col not in df_all.columns:
                df_all[col] = 0

        x_node = df_all.drop(columns=gnn_edge_fields, errors='ignore')
        x_node = x_node.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
        x_tensor = torch.tensor(x_node.values, dtype=torch.float32)

        edge_index = torch.empty((2, 0), dtype=torch.int32)
        for f in gnn_edge_fields:
            if f in df_all.columns:
                edge_index = torch.cat([edge_index, create_edges(f, df_all)], dim=1)

        y_tensor = torch.cat([torch.tensor(y.values, dtype=torch.float32), torch.tensor([0.0])])
        data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

        gnn_model.eval()
        with torch.no_grad():
            out = gnn_model(data).squeeze()
            prob = float(out[-1].item())
            pred = int(prob > 0.5)

        return jsonify({
            "model_used": "fraudgnn",
            "prediction": pred,
            "fraud_probability": round(prob, 4)
        })

    except Exception as e:
        print("GNN Context Prediction Error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
from collections import deque

# In-memory buffer to simulate stream
stream_buffer = deque(maxlen=100)

@app.route('/stream', methods=['POST'])
def receive_streamed_txn():
    try:
        txn = request.get_json(force=True)
        # print(f"Received txn: {txn}")  
        stream_buffer.appendleft(txn)
        return jsonify({"status": "received", "buffer_size": len(stream_buffer)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_stream', methods=['GET'])
def get_streamed_txns():
    # print(f"Sending {len(stream_buffer)} records")  
    return jsonify(list(stream_buffer))


@app.route("/record_stream", methods=["POST"])
def record_stream():
    try:
        payload = request.get_json(force=True)

        # Auto-tag type if not specified
        if "type" not in payload:
            if "card1" in payload:
                payload["type"] = "tree"
            else:
                payload["type"] = "gnn"

        stream_buffer.append(payload)
        return jsonify({"status": "recorded", "count": len(stream_buffer)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_stream", methods=["GET"])
def get_stream():
    try:
        return jsonify(list(stream_buffer))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === Run Flask Server ===
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
