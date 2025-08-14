""" 
tabpfn_final.py
Author: Aylin Ergun
v2025.8.14

Trains and uses TabPFN to make binary predictions (BRS3 vs BRS1/2) of 
BRS response subtype on the CHIMERA Grand Challenge Task 2 clinical data dataset.

Uses a KFold run (10 folds by default) to train and make predictions over the whole dataset. 
Resulting scores and label predictions are saved to a csv.
"""

def warn(*args, **kwargs): # sklearn doesn't like that feature names get stripped in feature selection, this is to suppress that warning
    pass
import warnings
warnings.warn = warn

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, chi2

from tabpfn import TabPFNClassifier

from pathlib import Path
import argparse
import random as rand
import pandas as pd
import numpy as np


def tabpfn_predict(training_data_path, output_name):
    """
    Trains TabPFN in subsequent folds of a KFold, predicting on the fold left out,
    iterating over the entire dataset in totality. Top 50% of features in the
    clinical data are selected for prediciton. The resulsts are de-shuffled and saved to csv.

    Inputs
        training_data_path: str, path to the clinical data csv
        output_name: str, filename for the results csv

    Outputs
        <output_name>.csv in the parent folder containing resutlts
    """

    training_data_path = Path(training_data_path) # Input validation
    if not training_data_path.is_file():
        print(f"Error: File not found at {training_data_path}")
        return
    
    # Load data
    df = pd.read_csv(training_data_path)
    X = df.drop(columns=["BRS3", "patient_id"])
    y = df["BRS3"]

    X[X < 0] = 0 # Feature selection doesn't like negative values

    # Cross validation
    FOLD_COUNT = 10
    random_state = rand.randint(0,100) # Effectively none, change this assignment to a specific value for reproducability
    kf = KFold(n_splits=FOLD_COUNT, shuffle=True, random_state=random_state)
    accuracy_list = []
    auroc_list = []
    results = []

    for train, test in kf.split(df):

        # Data split
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]

        # Feature selection
        fselect = SelectKBest(chi2, k=12)
        X_train_new = fselect.fit_transform(X_train, y_train)
        X_train_new = pd.DataFrame(X_train_new)

        X_test_new = X_test.loc[:, fselect.get_support()] # makes X_test_new only contain the features selected from the training set

        # Initialize a classifier
        clf = TabPFNClassifier()
        clf.fit(X_train_new, y_train)

        # Prediction
        prediction_probabilities = clf.predict_proba(X_test_new)
        predictions = clf.predict(X_test_new)

        # Match predictions to entries in the whole dataset
        for idx, test_idx in enumerate(X_test.index):
            results.append({
                "slide_id": df.loc[test_idx, "patient_id"],
                "label": y_test.iloc[idx],
                "probability": prediction_probabilities[idx, 1],
                "prediction": predictions[idx]})

        auroc = roc_auc_score(y_test, prediction_probabilities[:, 1])
        accuracy = accuracy_score(y_test, predictions)
        auroc_list.append(auroc)
        accuracy_list.append(accuracy)

    # Sort samples by patient ID
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="slide_id")

    # Model performance metrics
    auroc_avg = np.mean(auroc_list)
    auroc_sd = np.std(auroc_list)
    accuracy_avg = np.mean(accuracy_list)
    accuracy_sd = np.std(accuracy_list)

    print("Fold count:", FOLD_COUNT)
    print("Average AUROC:", auroc_avg)
    print("AUROC SD:", auroc_sd)
    print("Average Accuracy:", accuracy_avg)
    print("Accuracy SD:", accuracy_sd)
    print(classification_report(results_df["label"], results_df["prediction"]))
        
    # Save to csv
    results_df.to_csv(output_name, index=False)
    print("Saved scores and predictions to ", output_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run TabPFN to make a KFold cross validated prediction on the given dataset."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input data file. Will be split into training and test sets automatically."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="tabpfn_pred_probs.csv",
        help="Name for the output CSV file containing scores and predictions per sample."
    )
    args = parser.parse_args()

    tabpfn_predict(args.input_file, args.output_csv)
