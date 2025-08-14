"""
integrator_mean.py
Author: Aylin Ergun
v2025.8.14

Takes the average of the scores given by the clinical and imaging models, and makes a final classification.

Takes in paths containing pre_prepared predictions for now. Will later be fed predictions from clin and image models automatically.
"""

import pandas as pd
from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def integrate_mean(clin_data, img_data, output_csv):

    clin_data = Path(clin_data) # Input validation
    if not clin_data.is_file():
        print(f"Error: File not found at {clin_data}")
        return
    img_data = Path(img_data)
    if not img_data.is_file():
        print(f"Error: File not found at {img_data}")
        return
    
    # Load data and merge
    clin_df = pd.read_csv(clin_data)
    img_df = pd.read_csv(img_data)
    merged_df = pd.merge(clin_df, img_df, on="slide_id", suffixes=("_tabpfn", "_titan")) # Change suffixes if models change

    # Clean redundant columns
    merged_df = merged_df.drop(columns={"case_id", "label_titan"})
    merged_df = merged_df.rename(columns={"label_tabpfn": "label"})

    # Average the probabilities, make classifications
    merged_df["mean_probability"] = merged_df[["probability_tabpfn", "probability_titan"]].mean(axis=1)
    mean_score = merged_df["mean_probability"]
    THRESHOLD = 0.5
    merged_df["final_prediction"] = merged_df["mean_probability"] > THRESHOLD
    final_prediction = merged_df["final_prediction"]

    y = merged_df["label"]
    auroc = roc_auc_score(y, mean_score)
    accuracy = accuracy_score(y, final_prediction)

    print(f"AUROC: {auroc}")
    print(f"Accuracy: {accuracy}")
    print(classification_report(y, final_prediction))

    # Save to csv
    merged_df.to_csv(output_csv, index=False)
    print("Saved final results to ", output_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Integrate the scores of the clinical and image predictions and make a final classification based on the averages."
    )
    parser.add_argument(
        "--clin_path",
        type=str,
        required=True,
        help="Path to the file containing predictions made on clinical data."
    )
    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Path to the file containing predictions made on imaging data."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="final_results.csv",
        help="Name for the output CSV file containing final scores and predictions per sample."
    )
    args = parser.parse_args()

    integrate_mean(args.clin_path, args.img_path, args.output_csv)
