import requests
import json

# URL for the GNN prediction endpoint
gnn_url = "http://127.0.0.1:5000/predict_gnn"

# Hand-crafted suspicious transaction (values are exaggerated to simulate fraud)
known_fraud_case = {
    "TransactionAmt": 99999.99,
    "TransactionDT": 172800,
    "card3": 255,
    "card4": "discover",
    "C5": 5.0,
    "C6": 4.5,
    "C8": 6.0,
    "C9": 7.0,
    "ProductCD": "W",
    "DeviceInfo": "jailbroken-rooted-hackOS",
    "id_01": 9.9,
    "id_02": 8.8,
    "id_16": "not_found",
    "id_31": "bot_browser",
    "id_36": 1.0,
    "V9": 0.99,
    "V46": 0.98,
    "V51": 0.97,
    "V98": 0.96,
    "V103": 0.95,
    "V115": 0.94,
    "V132": 0.93,
    "V134": 0.92,
    "V172": 0.91,
    "V180": 0.90,
    "V183": 0.89,
    "V184": 0.88,
    "V185": 0.87,
    "V204": 0.86,
    "V210": 0.85,
    "V213": 0.84,
    "V224": 0.83,
    "V228": 0.82,
    "V232": 0.81,
    "V233": 0.80,
    "V235": 0.79,
    "V239": 0.78,
    "V252": 0.77,
    "V258": 0.76,
    "V261": 0.75,
    "V276": 0.74,
    "V291": 0.73,
    "V302": 0.72,
    "V319": 0.71,
    "V320": 0.70,
}

# Send request to GNN API
response = requests.post(gnn_url, json=known_fraud_case)

# Print result
print("\n=== Known Fraud Test Case (High-Risk Input) ===")
if response.status_code == 200:
    print("Prediction Response:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Error {response.status_code}: {response.text}")
