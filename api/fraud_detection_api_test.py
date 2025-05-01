import requests
import json
import random
from datetime import datetime

# This is the URL where the Flask API is running
url = "http://127.0.0.1:5000/predict_full"

# We're only testing traditional ML models here (not autoencoder or GNN)
models_to_test = [
    "randomforest",
    "xgboost",
    "logisticregression",
    "gradientboosting",
    "mlp"
]

# This function creates a realistic-looking, everyday transaction
# It simulates a typical purchase by a customer using normal feature values
def generate_normal_transaction():
    transaction = {
        "TransactionAmt": round(random.uniform(5, 500), 2),
        "card1": random.randint(1000, 8000),
        "card2": random.randint(100, 600),
        "card3": random.randint(100, 250),
        "card4": random.choice(["visa", "mastercard"]),
        "card5": random.randint(100, 300),
        "card6": random.choice(["debit", "credit"]),
        "addr1": random.randint(100, 500),
        "addr2": random.randint(50, 100),
        "dist1": random.uniform(0, 50),
        "dist2": random.uniform(0, 20),
        "TransactionDT": random.randint(86400, 172800),
        "P_emaildomain": random.choice(["gmail.com", "yahoo.com"]),
        "R_emaildomain": random.choice(["gmail.com", "yahoo.com"]),
        "DeviceType": random.choice(["desktop", "mobile"])
    }

    # Add more features that models might expect (like count and distance values)
    for i in range(1, 15):
        transaction[f"C{i}"] = random.randint(0, 2)
    for i in range(1, 16):
        transaction[f"D{i}"] = random.randint(0, 30)
    for i in range(1, 340):
        transaction[f"V{i}"] = round(random.uniform(0, 0.5), 4)

    return transaction

# This one simulates a high-risk or suspicious transaction.
# Values here are deliberately out-of-the-ordinary to mimic fraud behavior.
def generate_fraud_like_transaction():
    transaction = {
        "TransactionAmt": round(random.uniform(10000, 100000), 2),
        "card1": random.randint(9000, 999999),
        "card2": random.randint(100, 600),
        "card3": random.randint(100, 250),
        "card4": random.choice(["visa", "mastercard", "american express", "discover"]),
        "card5": random.randint(100, 300),
        "card6": random.choice(["debit", "credit"]),
        "addr1": random.randint(100, 500),
        "addr2": random.randint(50, 100),
        "dist1": random.uniform(0, 500),
        "dist2": random.uniform(0, 100),
        "TransactionDT": random.randint(86400, 172800),
        "P_emaildomain": random.choice(["gmail.com", "yahoo.com", "hotmail.com", "aol.com"]),
        "R_emaildomain": random.choice(["gmail.com", "yahoo.com", "hotmail.com", "aol.com"]),
        "DeviceType": random.choice(["desktop", "mobile", "toaster", "tablet"])
    }

    # Inject wider and more extreme values to make it look suspicious
    for i in range(1, 15):
        transaction[f"C{i}"] = random.randint(0, 5)
    for i in range(1, 16):
        transaction[f"D{i}"] = random.randint(0, 100)
    for i in range(1, 340):
        transaction[f"V{i}"] = round(random.uniform(0, 1), 4)

    return transaction

# === Main testing loop ===
for model_name in models_to_test:
    # Decide randomly whether to test a fraud-like or normal transaction
    if random.random() < 0.3:
        transaction = generate_fraud_like_transaction()
        transaction_type = "fraud-like"
    else:
        transaction = generate_normal_transaction()
        transaction_type = "normal"

    # Tell the API which model to use
    transaction["model"] = model_name

    try:
        # Send transaction to the Flask API
        response = requests.post(url, json=transaction)
        response.raise_for_status()

        # Get the API response
        result = response.json()

        # Print out the results so we can inspect them
        print(f"\n=== Testing model: {model_name} ({transaction_type}) ===")
        print("Status Code:", response.status_code)
        print("JSON Response:")
        print(json.dumps(result, indent=4))

        # Save the response to a file in case we want to compare later
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"output_{model_name}_{transaction_type}_{timestamp}.txt"
        with open(file_name, "w") as f:
            f.write(f"=== Model: {model_name} ({transaction_type}) ===\n")
            f.write(f"Status Code: {response.status_code}\n")
            f.write(json.dumps(result, indent=4))
            f.write("\n")

    except Exception as e:
        print(f"\n=== Testing model: {model_name} ({transaction_type}) ===")
        print(f"Failed to get prediction for {model_name}!")
        print("Error:", str(e))
