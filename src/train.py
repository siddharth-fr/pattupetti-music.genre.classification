"""
Machine Learning Model Training Module for Music Genre Classification

Trains and compares 6 traditional classifiers:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM with RBF kernel)
6. Gaussian Naive Bayes

Features:
- Stratified 80/20 Train-Test split.
- Proper feature scaling (StandardScaler fit strictly on train split to prevent data leakage).
- 5-Fold Stratified Cross-Validation on the training set.
- Macro F1-score model selection.
- Persistence of best model, scaler, and feature list using Joblib.
"""

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# The 6 Syllabus Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def get_models() -> dict:
    """
    Returns a dictionary of initialized baseline classifiers.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Support Vector Machine (SVM)": SVC(kernel="rbf", probability=True, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
    }


def load_and_preprocess_data(csv_path: str = "data/processed/features_30s.csv", test_size: float = 0.2, random_state: int = 42):
    """
    Loads extracted features, separates target and features,
    performs stratified train/test split, and applies feature scaling without leakage.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    df = pd.read_csv(csv_path)
    
    # Drop identifier columns to isolate numerical features
    non_feature_cols = ["track_id", "genre", "file_path"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    X = df[feature_cols].copy()
    y = df["genre"].copy()

    # Impute any rare NaN or Inf values if present
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature Scaling: Fit strictly on training set, then transform test set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def train_and_compare_models(csv_path: str = "data/processed/features_30s.csv", models_dir: str = "models"):
    """
    Runs cross-validation and test set evaluation across all 6 models,
    selects the highest Macro F1 performer, and saves the artifacts.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(" 1. Loading and Splitting Feature Dataset")
    print("=" * 65)
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(csv_path)

    print(f"Training samples : {len(y_train)} ({len(np.unique(y_train))} balanced genres)")
    print(f"Testing samples  : {len(y_test)}")
    print(f"Total features   : {len(feature_names)}")

    models = get_models()
    results = []

    print("\n" + "=" * 65)
    print(" 2. 5-Fold Stratified Cross-Validation & Model Training")
    print("=" * 65)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    for name, model in models.items():
        print(f"\n--> Training: {name}...")
        start_time = time.time()

        # 5-Fold Cross Validation for robust syllabus-standard comparison
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        cv_f1_mean = np.mean(cv_scores)
        cv_f1_std = np.std(cv_scores)

        # Train on full training set
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate on unseen test set
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        test_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "Model": name,
            "CV F1 (Mean)": round(cv_f1_mean, 4),
            "CV F1 (Std)": round(cv_f1_std, 4),
            "Test Accuracy": round(test_acc, 4),
            "Test Precision": round(test_prec, 4),
            "Test Recall": round(test_rec, 4),
            "Test Macro F1": round(test_f1, 4),
            "Train Time (s)": round(train_time, 2)
        })

        print(f"    5-Fold CV Macro F1 : {cv_f1_mean:.4f} (+/- {cv_f1_std:.4f})")
        print(f"    Test Accuracy      : {test_acc:.4f}")
        print(f"    Test Macro F1      : {test_f1:.4f}  (Time: {train_time:.2f}s)")

        # Track the best model based on Test Macro F1
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_name = name
            best_model_obj = model

    df_results = pd.DataFrame(results).sort_values(by="Test Macro F1", ascending=False)
    
    print("\n" + "=" * 65)
    print(" 3. Final Model Comparison Table")
    print("=" * 65)
    print(df_results.to_string(index=False))

    # Save results table
    results_csv = models_path / "model_comparison.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\nSaved comparison table to: {results_csv}")

    # Save best model and preprocessing scaler
    print("\n" + "=" * 65)
    print(f" 4. Saving Best Model: {best_model_name} (Macro F1 = {best_f1:.4f})")
    print("=" * 65)
    
    joblib.dump(best_model_obj, models_path / "best_model.joblib")
    joblib.dump(scaler, models_path / "scaler.joblib")
    joblib.dump(feature_names, models_path / "feature_names.joblib")
    joblib.dump(best_model_name, models_path / "best_model_name.joblib")

    print(f"Saved artifacts to {models_path}/:")
    print("  - best_model.joblib")
    print("  - scaler.joblib")
    print("  - feature_names.joblib")
    print("  - best_model_name.joblib")

    return df_results, best_model_name, best_model_obj, X_test, y_test


if __name__ == "__main__":
    train_and_compare_models()

