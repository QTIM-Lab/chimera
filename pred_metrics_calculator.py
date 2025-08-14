"""
pred_metrics_calculator.py
Author: Aylin Ergun
v2025.8.6

Short script to calculate and print performance metrics for a set of predictions in csv format.
"""

from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pandas as pd

def calculate_performance_metrics(input_path):

    input_path = Path(input_path) # Input validation
    if not input_path.is_file():
        print(f"Error: File not found at {input_path}")
        return

    # Load data
    df = pd.read_csv(input_path)
    labels = df["label"]
    predictions = df["prediction"]
    scores = df["probability"]

    # Calculate and print metrics
    print("AUROC:", roc_auc_score(labels, scores))
    print("Accuracy", accuracy_score(labels, predictions))
    print(classification_report(labels, predictions))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate AUROC, accuracy, F1 score for the given set of predictions and true labels."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input data file."
    )
    args = parser.parse_args()

    calculate_performance_metrics(args.input_file)
