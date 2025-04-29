from pathlib import Path
from data_loader import load_raw_data, merge_data
from preprocessing import drop_high_missing, build_preprocessor
from utils import frequency_encode, add_missing_flags
from training import train_and_evaluate

def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    transaction, identity = load_raw_data(raw_dir)
    df = merge_data(transaction, identity)

    df = drop_high_missing(df)
    df = add_missing_flags(df)

    # Frequency encode categorical columns with < 500 unique values
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = frequency_encode(df, cat_cols, max_unique=500)

    target = 'isFraud'
    features_to_drop = ['TransactionID', target]
    X = df.drop(columns=features_to_drop)
    y = df[target]

    preprocessor, _, _ = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    train_and_evaluate(X_processed, y, preprocessor, processed_dir)

if __name__ == "__main__":
    main()
