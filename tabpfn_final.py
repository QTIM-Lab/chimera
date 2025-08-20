""" 
tabpfn_final.py
Author: Aylin Ergun (extended by Junie)
v2025.8.20

Trains and uses TabPFN to make binary predictions (BRS3 vs BRS1/2) on
CHIMERA Task 2 clinical dataset.

Enhancements for Grand Challenge submission:
- Training on full labeled dataset and saving a serialized model bundle (feature selector + TabPFN + schema)
- Loading a saved model for inference on unlabeled CSVs
- Robust handling of NaNs and negative values for chi2 feature selection
- Dynamic feature count cap and CPU-only execution for container safety
- Backward-compatible CV evaluation function
- Added SMOTE to handle class imbalance during training and CV.
- Added Feature Engineering to create more informative features.
"""
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, chi2

from tabpfn import TabPFNClassifier
from imblearn.over_sampling import SMOTE

from pathlib import Path
import argparse
import os
import random as rand
import pandas as pd
import numpy as np
import joblib
from typing import List, Tuple, Dict, Any

try:
    import torch
except Exception:
    torch = None


# -----------------------------
# Utility helpers
# -----------------------------

def set_global_seed(seed: int) -> None:
    """Set seeds for python, numpy, and torch (if available) to improve reproducibility."""
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    try:
        rand.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
            if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass

# <<< --- FEATURE ENGINEERING START --- >>>
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features from the existing data to improve model performance.
    """
    # Make a copy to avoid modifying the original DataFrame
    df_eng = df.copy()

    # 1. Binning 'age'
    # Creates categorical groups for age
    if 'age' in df_eng.columns:
        age_bins = [0, 60, 75, 120]
        age_labels = ['age_lt_60', 'age_60_75', 'age_gt_75']
        df_eng['age_group'] = pd.cut(df_eng['age'], bins=age_bins, labels=age_labels, right=False)
        # One-hot encode the new age groups
        df_eng = pd.get_dummies(df_eng, columns=['age_group'], prefix='age')

    # 2. Binning 'no_instillations'
    # Groups the number of instillations, treating -1 as a separate category
    if 'no_instillations' in df_eng.columns:
        inst_bins = [-2, 0, 10, 20, 100]
        inst_labels = ['inst_unknown', 'inst_1_10', 'inst_11_20', 'inst_gt_20']
        df_eng['inst_group'] = pd.cut(df_eng['no_instillations'], bins=inst_bins, labels=inst_labels, right=True)
        # One-hot encode the new instillation groups
        df_eng = pd.get_dummies(df_eng, columns=['inst_group'], prefix='inst')

    # 3. Interaction Feature: 'female' and 'smoker'
    # Creates a feature that is 1 only if the patient is a female smoker
    if 'female' in df_eng.columns and 'smoker' in df_eng.columns:
        df_eng['female_smoker'] = df_eng['female'].astype(int) * df_eng['smoker'].astype(int)

    return df_eng
# <<< --- FEATURE ENGINEERING END --- >>>

def _ensure_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only numeric columns, coerce others if present
    num_df = df.apply(pd.to_numeric, errors="coerce")
    # Fill NaNs with 0 and ensure non-negative for chi2
    num_df = num_df.fillna(0)
    num_df[num_df < 0] = 0
    return num_df


def _align_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    # Ensure all training-time columns exist in test df; missing -> 0
    aligned = pd.DataFrame({col: (df[col] if col in df.columns else 0) for col in feature_columns})
    # Ensure numeric, non-negative
    aligned = _ensure_numeric_df(aligned)
    return aligned


# -----------------------------
# Core functions
# -----------------------------

def tabpfn_predict(training_data_path: str, output_name: str, k_features: int = 12, seed: int = 42) -> None:
    """
    Perform 10-fold CV over the entire labeled dataset and save out-of-fold predictions to CSV.
    """
    set_global_seed(int(seed))
    training_data_path = Path(training_data_path)
    if not training_data_path.is_file():
        print(f"Error: File not found at {training_data_path}")
        return

    # Load dataset
    df_raw = pd.read_csv(training_data_path)

    # Apply Feature Engineering
    df = engineer_features(df_raw)

    if "BRS3" not in df.columns or "patient_id" not in df.columns:
        print("Error: Input CSV must contain 'patient_id' and 'BRS3' columns for CV mode.")
        return

    X = df.drop(columns=["BRS3", "patient_id"]) if df.shape[1] > 2 else pd.DataFrame(index=df.index)
    X = _ensure_numeric_df(X)
    y = df["BRS3"].astype(int)

    # Cross validation
    FOLD_COUNT = 10
    kf = KFold(n_splits=FOLD_COUNT, shuffle=True, random_state=int(seed))
    results = []
    auroc_list = []

    fold_counter = 0
    for train_idx, test_idx in kf.split(df):
        fold_counter += 1
        print(f"--- Processing Fold {fold_counter}/{FOLD_COUNT} ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        n_feats = X_train.shape[1]
        k = max(1, min(k_features, n_feats)) if n_feats > 0 else 1
        fselect = SelectKBest(chi2, k=k)
        X_train_new = fselect.fit_transform(X_train, y_train)
        selected_features = X_train.columns[fselect.get_support()]
        X_test_new = X_test[selected_features]

        print(f"Original training distribution: {y_train.value_counts().to_dict()}")
        smote = SMOTE(random_state=int(seed))
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_new, y_train)
        print(f"Resampled training distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")

        clf = TabPFNClassifier(device="cpu")
        clf.fit(X_train_resampled, y_train_resampled)

        prediction_probabilities = clf.predict_proba(X_test_new)
        predictions = clf.predict(X_test_new)

        for i, row_idx in enumerate(X_test.index):
            results.append(
                {
                    "slide_id": df.loc[row_idx, "patient_id"],
                    "label": int(y_test.iloc[i]),
                    "probability": float(prediction_probabilities[i, 1]),
                    "prediction": int(predictions[i]),
                }
            )
        try:
            auroc = roc_auc_score(y_test, prediction_probabilities[:, 1])
            auroc_list.append(auroc)
        except Exception:
            pass

    results_df = pd.DataFrame(results).sort_values(by="slide_id")
    auroc_avg = float(np.mean(auroc_list)) if len(auroc_list) else float("nan")

    print("\n--- CV Results ---")
    print("Average AUROC:", auroc_avg)
    try:
        print("\nClassification Report (on out-of-fold predictions):")
        print(classification_report(results_df["label"], results_df["prediction"]))
    except Exception:
        pass

    results_df.to_csv(output_name, index=False)
    print("Saved scores and predictions to", output_name)


def train_and_save_model(training_data_path: str, save_model_path: str, k_features: int = 12, seed: int = 42) -> None:
    """Train TabPFN on the full labeled dataset and save the selector+model+schema."""
    set_global_seed(int(seed))
    training_data_path = Path(training_data_path)
    if not training_data_path.is_file():
        raise FileNotFoundError(f"Input file not found: {training_data_path}")

    df_raw = pd.read_csv(training_data_path)
    # Apply Feature Engineering
    df = engineer_features(df_raw)

    if "BRS3" not in df.columns or "patient_id" not in df.columns:
        raise ValueError("Input CSV must contain 'patient_id' and 'BRS3' columns for training.")

    feature_columns = [c for c in df.columns if c not in ("BRS3", "patient_id")]
    X = _ensure_numeric_df(df[feature_columns]) if feature_columns else pd.DataFrame(index=df.index)
    y = df["BRS3"].astype(int)

    n_feats = X.shape[1]
    k = max(1, min(k_features, n_feats)) if n_feats > 0 else 1
    fselect = SelectKBest(chi2, k=k)
    X_selected = fselect.fit_transform(X, y)

    print(f"Original training distribution: {y.value_counts().to_dict()}")
    smote = SMOTE(random_state=int(seed))
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)
    print(f"Resampled training distribution: {pd.Series(y_resampled).value_counts().to_dict()}")

    clf = TabPFNClassifier(device="cpu")
    clf.fit(X_resampled, y_resampled)

    bundle = {
        "version": "2025-08-20",
        "feature_columns": feature_columns,
        "selector": fselect,
        "clf": clf,
        "k_features": k,
    }
    save_model_path = Path(save_model_path)
    save_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_model_path)
    print(f"Model saved to {save_model_path} (k_features={k}, n_features={n_feats})")


def predict_with_saved_model(test_data_path: str, model_path: str, output_csv: str, seed: int = 42) -> None:
    """Load a saved bundle and run inference on input CSV which may not contain labels."""
    set_global_seed(int(seed))
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    test_data_path = Path(test_data_path)
    if not test_data_path.is_file():
        raise FileNotFoundError(f"Input file not found: {test_data_path}")

    bundle: Dict[str, Any] = joblib.load(model_path)
    feature_columns: List[str] = bundle["feature_columns"]
    selector = bundle["selector"]
    clf = bundle["clf"]

    df_raw = pd.read_csv(test_data_path)
    # Apply the same feature engineering
    df = engineer_features(df_raw)

    id_col = "patient_id" if "patient_id" in df.columns else ("slide_id" if "slide_id" in df.columns else None)
    if id_col is None:
        raise ValueError("Input CSV must contain an identifier column: 'patient_id' or 'slide_id'.")

    X_raw = df.drop(columns=[c for c in ["BRS3", id_col] if c in df.columns])
    X_aligned = _align_features(X_raw, feature_columns)
    X_sel = selector.transform(X_aligned)

    probs = clf.predict_proba(X_sel)
    preds = clf.predict(X_sel)

    labels = df["BRS3"].astype(int).tolist() if "BRS3" in df.columns else [-1] * len(df)

    results = pd.DataFrame(
        {
            "slide_id": df[id_col].values,
            "label": labels,
            "probability": probs[:, 1],
            "prediction": preds,
        }
    ).sort_values(by="slide_id")
    results.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabPFN utilities for CHIMERA (CV/train/predict)")
    parser.add_argument("--mode", type=str, choices=["cv", "train", "predict"], default="cv")
    parser.add_argument("--input_file", type=str, help="Path to input CSV")
    parser.add_argument("--output_csv", type=str, default="tabpfn_pred_probs.csv", help="Output CSV path")
    parser.add_argument("--k_features", type=int, default=12, help="Number of features to select (upper bound)")
    parser.add_argument("--save_model_path", type=str, help="Path to save trained model (train mode)")
    parser.add_argument("--model_path", type=str, help="Path to load model for prediction (predict mode)")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducibility")
    args = parser.parse_args()

    if args.mode == "cv":
        if not args.input_file:
            raise SystemExit("--input_file is required for cv mode")
        tabpfn_predict(args.input_file, args.output_csv, k_features=args.k_features, seed=args.seed)
    elif args.mode == "train":
        if not args.input_file or not args.save_model_path:
            raise SystemExit("--input_file and --save_model_path are required for train mode")
        train_and_save_model(args.input_file, args.save_model_path, k_features=args.k_features, seed=args.seed)
    elif args.mode == "predict":
        if not args.input_file or not args.model_path:
            raise SystemExit("--input_file and --model_path are required for predict mode")
        predict_with_saved_model(args.input_file, args.model_path, args.output_csv, seed=args.seed)
