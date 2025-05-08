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

# === Resolve project root ===
project_root = Path(__file__).resolve()
while not (project_root / "gnn").exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.append(str(project_root))
print("Project root set to:", project_root)

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

# === Init Flask app ===
app = Flask(__name__)

# === Paths ===
manual_model_dir = project_root / "models" / "trained_model_manual_fields"
stream_model_dir = project_root / "models" / "trained_model_selected_subset"
processed_dir = project_root / "data" / "processed"

# === Feature Lists ===
manual_features = ["TransactionAmt", "card1", "addr1", "dist1", "ProductCD_freq", "P_emaildomain_freq"]

stream_features = [
    "TransactionAmt", "TransactionDT", "card1", "card4_freq", "card6_freq", "addr1", "dist1",
    "P_emaildomain_freq", "R_emaildomain_freq", "M1_freq", "M4_freq", "M5_freq", "M6_freq", "M9_freq",
    "C1", "C2", "C8", "C11", "V18", "V21", "V97", "V133", "V189", "V200", "V258", "V282", "V294", "V312",
    "DeviceType_freq", "id_15_freq", "id_28_freq", "id_29_freq", "id_31_freq", "id_35_freq", "id_36_freq",
    "id_37_freq", "id_38_freq"
]

gnn_features = [
    'V233', 'V132', 'C5', 'V261', 'V134', 'V302', 'DeviceInfo', 'V184', 'V183', 'V239', 'V224', 'V291',
    'id_31', 'V228', 'V319', 'V185', 'id_16', 'V235', 'V258', 'id_01', 'V213', 'V9', 'V98', 'V320', 'V232',
    'id_02', 'V204', 'V115', 'V172', 'V252', 'V276', 'V210', 'V180', 'id_36', 'V46', 'V51', 'V103',
    'TransactionDT', 'TransactionAmt', 'ProductCD', 'card4', 'C8', 'C9', 'card3', 'C6'
]

gnn_edge_fields = ['card1', 'addr1', 'addr2', 'P_emaildomain', 'DeviceType', 'id_17', 'id_28']

# === Load Preprocessors and Frequency Maps ===
manual_preprocessor = joblib.load(manual_model_dir / "preprocessor_manual.pkl")
stream_preprocessor = joblib.load(stream_model_dir / "preprocessor.pkl")
freq_maps = joblib.load(processed_dir / "frequency_maps.pkl")
product_freq = freq_maps.get("ProductCD", {})
email_freq = freq_maps.get("P_emaildomain", {})

# === Load Tree-Based Models ===
manual_models = {name: joblib.load(manual_model_dir / f"{name}_manual_model.pkl") for name in ["randomforest", "xgboost", "logisticregression", "gradientboosting", "mlp"]}
stream_models = {name: joblib.load(stream_model_dir / f"{name}_selected_model.pkl") for name in ["randomforest", "xgboost", "logisticregression", "gradientboosting", "mlp"]}

# === Load GNN Model from .pt ===
# === Load GNN model structure from .pkl and then load weights from .pt ===


gnn_model_pkl_path = project_root / "gnn" / "fraudgnn_model.pkl"
gnn_model_weights_path = project_root / "gnn" / "best_fraudgnn_model.pt"

with open(gnn_model_pkl_path, "rb") as f:
    gnn_model = pickle.load(f)

# Ensure consistency by loading fresh weights
gnn_state_dict = torch.load(gnn_model_weights_path, map_location="cpu")
gnn_model.load_state_dict(gnn_state_dict)
gnn_model.eval()

# === SHAP Explainers for Tree Models ===
manual_explainers, stream_explainers = {}, {}
for name in ["xgboost", "randomforest"]:
    try:
        manual_explainers[name] = shap.TreeExplainer(manual_models[name])
        stream_explainers[name] = shap.TreeExplainer(stream_models[name])
    except Exception as e:
        print(f"SHAP setup failed for {name}: {e}")

def create_edges(feature_column, df, max_edges_per_group=100):
    edge_list = []
    groups = df.groupby(feature_column).indices
    for _, indices in groups.items():
        if len(indices) < 2:
            continue
        pair_generator = combinations(indices, 2)
        limited_pairs = list(islice(pair_generator, max_edges_per_group))
        edge_list.extend(limited_pairs)
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.int32)
    return torch.tensor(edge_list, dtype=torch.int32).t().contiguous()

# === Tab 1 Manual Prediction Endpoint ===
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    data = request.get_json(force=True)
    model_name = data.get("model", "xgboost").lower()
    model = manual_models.get(model_name)
    if not model:
        return jsonify({"error": f"Model '{model_name}' not found"}), 400

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

    shap_out = {}
    if model_name in manual_explainers:
        try:
            shap_out = {f: float(v) for f, v in zip(manual_features, manual_explainers[model_name](X).values[0])}
        except:
            shap_out = {"warning": "SHAP unavailable"}

    return jsonify({
        "model_used": model_name,
        "input": data,
        "filled_input": df.iloc[0].to_dict(),
        "prediction": y_pred,
        "fraud_probability": round(prob, 4),
        "shap_top_contributors": shap_out
    })

# === Tab 2 Stream Prediction Endpoint ===
@app.route('/predict_stream', methods=['POST'])
def predict_stream():
    data = request.get_json(force=True)
    model_name = data.get("model", "xgboost").lower()
    model = stream_models.get(model_name)
    if not model:
        return jsonify({"error": f"Model '{model_name}' not found"}), 400

    df = pd.DataFrame([data]).reindex(columns=stream_features, fill_value=0)
    X = stream_preprocessor.transform(df)

    try:
        y_pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    shap_out = {}
    if model_name in stream_explainers:
        try:
            shap_out = {f: float(v) for f, v in zip(stream_features, stream_explainers[model_name](X).values[0])}
        except:
            shap_out = {"warning": "SHAP unavailable"}

    return jsonify({
        "model_used": model_name,
        "input": data,
        "filled_input": df.iloc[0].to_dict(),
        "prediction": y_pred,
        "fraud_probability": round(prob, 4),
        "shap_top_contributors": shap_out
    })


@app.route('/predict_gnn_with_context', methods=['POST'])
def predict_gnn_with_context():
    try:
        print("Loading reduced features and labels exactly as used in training...")
        X = pd.read_csv(project_root / "gnn" / "reduced_features.csv").tail(100000).reset_index(drop=True)
        y = pd.read_csv(project_root / "gnn" / "balanced_labels.csv").squeeze().tail(len(X)).reset_index(drop=True)

        print("Receiving new transaction...")
        new_txn = request.get_json(force=True)
        if not isinstance(new_txn, dict):
            return jsonify({"error": "Invalid input. Expecting JSON object"}), 400

        print("Appending transaction without transformation...")
        df_new = pd.DataFrame([new_txn])
        df_all = pd.concat([X, df_new], ignore_index=True)

        # Ensure all GNN features exist (no missing columns)
        for col in gnn_features:
            if col not in df_all.columns:
                df_all[col] = 0

        print("Dropping relational edge fields from node feature matrix...")
        x_node = df_all.drop(columns=gnn_edge_fields, errors='ignore')

        print("Building edge index using training-style relational logic...")
        edge_index = torch.empty((2, 0), dtype=torch.int32)
        for feature in gnn_edge_fields:
            if feature in df_all.columns:
                edges = create_edges(feature, df_all, max_edges_per_group=100)
                edge_index = torch.cat([edge_index, edges], dim=1)

        print(f"Total edges formed: {edge_index.shape[1]}")

        print("Preparing PyG Data object...")
        # Coerce all columns to float32, replacing non-convertible entries with 0.0
        print("Coercing all features to float32...")
        x_node = x_node.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)


        x_tensor = torch.tensor(x_node.values, dtype=torch.float32)

        y_tensor = torch.cat([torch.tensor(y.values, dtype=torch.float32), torch.tensor([0.0])])

        data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

        print("Running model inference...")
        gnn_model.eval()
        with torch.no_grad():
            out = gnn_model(data).squeeze()
            prob = float(out[-1].item())
            pred = int(prob > 0.5)

        print(f"Prediction: {pred} (probability: {prob:.4f})")
        return jsonify({
            "model_used": "fraudgnn",
            "prediction": pred,
            "fraud_probability": round(prob, 4)
        })

    except Exception as e:
        print("GNN Context Prediction Error:", e)
        return jsonify({"error": str(e)}), 500


# === Start Flask Server ===
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
