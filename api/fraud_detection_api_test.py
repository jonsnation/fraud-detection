import pandas as pd
import requests
import json

# === Paths to your data ===
features_path = "C:/Users/jonat/fraud-detection-project/gnn/reduced_features.csv"
labels_path = "C:/Users/jonat/fraud-detection-project/gnn/balanced_labels.csv"
url = "http://127.0.0.1:5000/predict_gnn_with_context"

# === Load data ===
X = pd.read_csv(features_path)
y = pd.read_csv(labels_path).squeeze()

# === Combine to allow filtering
df = X.copy()
df["isFraud"] = y

# === Sample 5 fraud and 5 non-fraud rows
fraud_samples = df[df["isFraud"] == 1].sample(5, random_state=42)
nonfraud_samples = df[df["isFraud"] == 0].sample(5, random_state=42)

# === Prepare test payloads
transactions = [("FRAUD", row.drop("isFraud").to_dict()) for _, row in fraud_samples.iterrows()]
transactions += [("NON-FRAUD", row.drop("isFraud").to_dict()) for _, row in nonfraud_samples.iterrows()]

# === Send and display results
print("=== Sending 10 Transactions (5 Fraud, 5 Non-Fraud) to GNN API ===")
for i, (label, txn) in enumerate(transactions, start=1):
    print(f"\n--- [{i}] {label} ---")
    response = requests.post(url, json=txn)
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"Error {response.status_code}: {response.text}")
