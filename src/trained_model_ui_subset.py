import os
import joblib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def train_on_selected_features(X_full, y, save_dir="trained_model_selected_subset", n_splits=5):
    selected_features = [
        # Transaction features
        "TransactionAmt", "TransactionDT",
        # Card features
        "card1", "card4_freq", "card6_freq",
        # Address & Distance
        "addr1", "dist1",
        # Email domains
        "P_emaildomain_freq", "R_emaildomain_freq",
        # Match flags
        "M1_freq", "M4_freq", "M5_freq", "M6_freq", "M9_freq",
        # Count-based
        "C1", "C2", "C8", "C11",
        # Vesta features
        "V18", "V21", "V97", "V133", "V189", "V200", "V258", "V282", "V294", "V312",
        # Identity/device
        "DeviceType_freq", "id_15_freq", "id_28_freq", "id_29_freq", "id_31_freq", "id_35_freq", "id_36_freq", "id_37_freq", "id_38_freq"
    ]

    X = X_full[selected_features]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, selected_features)
    ])

    X_processed = preprocessor.fit_transform(X)
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(preprocessor, Path(save_dir) / "preprocessor.pkl")

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    }

    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        print(f"\n=== Training {name} with {n_splits}-Fold Cross-Validation on Selected Features ===")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs, rocs = [], []

        for i, (train_idx, val_idx) in enumerate(kf.split(X_processed)):
            X_train, X_val = X_processed[train_idx], X_processed[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            start = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - start

            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            acc = accuracy_score(y_val, y_pred)
            roc = roc_auc_score(y_val, y_prob)
            accs.append(acc)
            rocs.append(roc)

            print(f"Fold {i+1}: Accuracy = {acc:.4f}, ROC AUC = {roc:.4f}, Time = {duration:.2f}s")
            print("Classification Report:")
            print(classification_report(y_val, y_pred, digits=4))

        # Train on full data
        model.fit(X_processed, y)
        joblib.dump(model, Path(save_dir) / f"{name.lower()}_selected_model.pkl")

        # Plot ROC
        y_full_prob = model.predict_proba(X_processed)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_full_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")

        print(f">>> {name} - Mean Accuracy: {np.mean(accs):.4f}, Mean ROC AUC: {np.mean(rocs):.4f}")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - SHAP-Selected Feature Subset")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "roc_curves_selected.png")
    plt.show()
