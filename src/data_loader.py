import pandas as pd
from pathlib import Path

def load_raw_data(raw_dir: Path):
    transaction_path = raw_dir / "train_transaction.csv"
    identity_path = raw_dir / "train_identity.csv"

    transaction = pd.read_csv(transaction_path, encoding="utf-8-sig")
    identity = pd.read_csv(identity_path, encoding="utf-8-sig")

    # Fix incorrect TransactionID column name if needed
    if '2TransactionID' in transaction.columns:
        transaction.rename(columns={'2TransactionID': 'TransactionID'}, inplace=True)

    # Strip all whitespace just in case
    transaction.columns = transaction.columns.str.strip()
    identity.columns = identity.columns.str.strip()

    print("Loaded transaction shape:", transaction.shape)
    print("Loaded identity shape:", identity.shape)
    print("Transaction columns:", transaction.columns.tolist())
    print("Identity columns:", identity.columns.tolist())

    return transaction, identity



def merge_data(transaction: pd.DataFrame, identity: pd.DataFrame):
    merged = pd.merge(transaction, identity, on="TransactionID", how="left")
    print("Merged data shape:", merged.shape)
    return merged


def load_smote_data(processed_dir):
    X_path = processed_dir / "X_processed.csv"
    y_path = processed_dir / "y_processed.csv"  # Corrected filename

    # Use read_csv instead of read_pickle
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    print("Loaded SMOTE data shapes:")
    print(f"X: {X.shape}, y: {y.shape}")
    return X, y