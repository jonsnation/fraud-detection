import pandas as pd
import requests
import time

tree_df = pd.read_csv(r"C:\Users\jonat\fraud-detection-project\data\processed\X_subset_features.csv")
gnn_df = pd.read_csv(r"C:\Users\jonat\fraud-detection-project\gnn\reduced_features.csv")

API_URL = "http://localhost:5000/stream"

min_len = min(len(tree_df), len(gnn_df))

for i in range(min_len):
    tree_record = tree_df.iloc[i].dropna().to_dict()
    gnn_record = gnn_df.iloc[i].dropna().to_dict()

    # Inject type field
    tree_record["type"] = "tree"
    gnn_record["type"] = "gnn"

    try:
        requests.post(API_URL, json=tree_record)
        print(f"[{i}] Tree record sent.")
    except Exception as e:
        print(f"[ERROR] Tree record {i}: {e}")

    try:
        requests.post(API_URL, json=gnn_record)
        print(f"[{i}] GNN record sent.")
    except Exception as e:
        print(f"[ERROR] GNN record {i}: {e}")

    time.sleep(1)
