import joblib
import time
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, accuracy_score


def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print("Applied SMOTE. New class balance:")
    print(y_balanced.value_counts())
    return X_balanced, y_balanced

def train_and_evaluate(X, y, preprocessor, save_dir: Path):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    }

    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        print(f"\n Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start
        print(f"{name} trained in {duration:.2f} seconds")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print(f"\n=== {name} Results ===")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

        # Save model
        joblib.dump(model, save_dir / f"{name.lower()}_model.pkl")

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    # Finalize plot
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Models")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Save preprocessor only
    joblib.dump(preprocessor, save_dir / "preprocessor.pkl")
    print("All models and preprocessor saved.")


def train_and_evaluate_cv(X, y, preprocessor, save_dir: Path, n_splits: int = 5):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        print(f"\n=== Training {name} with {n_splits}-Fold Cross-Validation ===")
        fold = 1
        fold_accuracies = []
        fold_roc_aucs = []

        model_save_dir = save_dir / name.lower()
        model_save_dir.mkdir(parents=True, exist_ok=True)  

        for train_idx, test_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

            # Train
            start = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - start
            print(f"Fold {fold}: {name} trained in {duration:.2f} seconds")

            # Predict
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            # Metrics
            acc = accuracy_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_prob)
            fold_accuracies.append(acc)
            fold_roc_aucs.append(roc_auc)

            # Full classification report
            report = classification_report(y_val, y_pred, digits=4)
            print(f"\n=== Fold {fold} Classification Report for {name} ===")
            print(report)

            # Save report to file
            report_path = model_save_dir / f"{name.lower()}_fold{fold}_report.txt"
            with open(report_path, "w") as f:
                f.write(f"Fold {fold} - {name}\n")
                f.write(report)
                f.write(f"\nAccuracy: {acc:.4f}\nROC AUC: {roc_auc:.4f}\n")

            fold += 1

        # Save final trained model on full data
        model.fit(X, y)
        joblib.dump(model, model_save_dir / f"{name.lower()}_full_model.pkl")

        # Average performance
        avg_acc = np.mean(fold_accuracies)
        avg_roc_auc = np.mean(fold_roc_aucs)
        print(f"\n>>> {name} - Mean Accuracy: {avg_acc:.4f} | Mean ROC AUC: {avg_roc_auc:.4f}")

        # Plot final ROC Curve (trained on full data)
        y_full_prob = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_full_prob)
        roc_auc_full = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_full:.3f})")

    # Plot ROC Curve details
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({n_splits}-Fold CV Models)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Save preprocessor
    joblib.dump(preprocessor, save_dir / "preprocessor.pkl")
    print("\nAll models, preprocessor, and classification reports saved.")



    