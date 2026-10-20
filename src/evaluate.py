"""
Evaluation and Visualization Module for Music Genre Classification

Generates college-report-ready charts:
1. Confusion Matrix Heatmap for the best model.
2. Model Comparison Bar Chart (Accuracy vs Macro F1).
3. Detailed Classification Report saved to reports/classification_report.txt.
"""

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import confusion_matrix, classification_report
from src.train import load_and_preprocess_data


def plot_confusion_matrix(y_true, y_pred, labels, output_path: str = "reports/confusion_matrix.png", title: str = "Confusion Matrix"):
    """
    Plots a normalized confusion matrix heatmap with percentage annotations.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Normalized Accuracy'}
    )
    plt.title(f"{title} (Normalized)", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Predicted Genre", fontsize=12, labelpad=10)
    plt.ylabel("True Genre", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix chart to: {output_path}")


def plot_model_comparison(comparison_csv: str = "models/model_comparison.csv", output_path: str = "reports/model_comparison.png"):
    """
    Plots a side-by-side bar chart comparing Test Accuracy and Macro F1 across all models.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(comparison_csv)
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35

    plt.bar(x - width/2, df["Test Accuracy"], width, label="Test Accuracy", color="#3b82f6", alpha=0.9)
    plt.bar(x + width/2, df["Test Macro F1"], width, label="Test Macro F1", color="#10b981", alpha=0.9)

    plt.xlabel("Machine Learning Model", fontsize=12, labelpad=10)
    plt.ylabel("Score (0 to 1)", fontsize=12, labelpad=10)
    plt.title("Model Performance Comparison (FMA Small)", fontsize=14, weight="bold", pad=12)
    plt.xticks(x, df["Model"], rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="lower right", framealpha=0.9)

    # Annotate bar values
    for i in range(len(df)):
        plt.text(i - width/2, df["Test Accuracy"].iloc[i] + 0.02, f"{df['Test Accuracy'].iloc[i]:.2f}", ha='center', fontsize=9)
        plt.text(i + width/2, df["Test Macro F1"].iloc[i] + 0.02, f"{df['Test Macro F1'].iloc[i]:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved model comparison chart to: {output_path}")


def generate_evaluation_reports(csv_path: str = "data/processed/features_30s.csv", models_dir: str = "models", reports_dir: str = "reports"):
    """
    Loads saved best model and generates full report suite.
    """
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    best_model = joblib.load(f"{models_dir}/best_model.joblib")
    scaler = joblib.load(f"{models_dir}/scaler.joblib")
    best_name = joblib.load(f"{models_dir}/best_model_name.joblib")

    # Load test split
    _, X_test_scaled, _, y_test, _, _ = load_and_preprocess_data(csv_path)

    y_pred = best_model.predict(X_test_scaled)
    unique_genres = sorted(list(np.unique(y_test)))

    # 1. Plot Confusion Matrix
    plot_confusion_matrix(
        y_test, y_pred, unique_genres,
        output_path=str(reports_path / "confusion_matrix.png"),
        title=f"Confusion Matrix - {best_name}"
    )

    # 2. Plot Model Comparison Bar Chart
    comp_csv = f"{models_dir}/model_comparison.csv"
    if Path(comp_csv).exists():
        plot_model_comparison(comp_csv, output_path=str(reports_path / "model_comparison.png"))

    # 3. Save Text Classification Report
    clf_rep = classification_report(y_test, y_pred, target_names=unique_genres)
    report_file = reports_path / "classification_report.txt"
    with open(report_file, "w") as f:
        f.write(f"Best Model: {best_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(clf_rep)

    print(f"\nClassification report saved to: {report_file}")
    print("\n" + clf_rep)


if __name__ == "__main__":
    generate_evaluation_reports()
