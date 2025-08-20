""" 
model_comparison_script.py
Author: Aylin Ergun (extended by Junie & Gemini)
v2025.8.20

Compares, trains, and uses multiple models (TabPFN, RandomForest, XGBoost, LightGBM)
to make binary predictions (BRS3 vs BRS1/2) on the CHIMERA Task 2 clinical dataset.

New Features:
- CV mode now automatically compares multiple models and prints a summary table.
- Train/Predict modes require a --model argument to specify which model to use.
- Retains feature engineering, SMOTE for imbalance, and robust data handling.

Required installations:
pip install pandas numpy scikit-learn tabpfn imbalanced-learn xgboost lightgbm tabulate
"""
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

from tabpfn import TabPFNClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from tabulate import tabulate

from pathlib import Path
import argparse
import os
import random as rand
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any

try:
    import torch
except Exception:
    torch = None


# -----------------------------
# Utility helpers
# -----------------------------

def set_global_seed(seed: int) -> None:
    """Set seeds for python, numpy, and torch (if available) to improve reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    rand.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

# <<< --- FEATURE ENGINEERING START --- >>>
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features from the existing data to improve model performance.
    """
    df_eng = df.copy()
    if 'age' in df_eng.columns:
        age_bins = [0, 60, 75, 120]
        age_labels = ['age_lt_60', 'age_60_75', 'age_gt_75']
        df_eng['age_group'] = pd.cut(df_eng['age'], bins=age_bins, labels=age_labels, right=False)
        df_eng = pd.get_dummies(df_eng, columns=['age_group'], prefix='age')
    if 'no_instillations' in df_eng.columns:
        inst_bins = [-2, 0, 10, 20, 100]
        inst_labels = ['inst_unknown', 'inst_1_10', 'inst_11_20', 'inst_gt_20']
        df_eng['inst_group'] = pd.cut(df_eng['no_instillations'], bins=inst_bins, labels=inst_labels, right=True)
        df_eng = pd.get_dummies(df_eng, columns=['inst_group'], prefix='inst')
    if 'female' in df_eng.columns and 'smoker' in df_eng.columns:
        df_eng['female_smoker'] = df_eng['female'].astype(int) * df_eng['smoker'].astype(int)
    return df_eng
# <<< --- FEATURE ENGINEERING END --- >>>

def _ensure_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    num_df[num_df < 0] = 0
    return num_df

def _align_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    aligned = pd.DataFrame({col: (df[col] if col in df.columns else 0) for col in feature_columns})
    return _ensure_numeric_df(aligned)

def get_model(model_name: str, seed: int):
    """Factory function to get a model instance."""
    if model_name == 'tabpfen':
        return TabPFNClassifier(device="cpu")
    if model_name == 'randomforest':
        return RandomForestClassifier(random_state=seed, n_estimators=100, class_weight='balanced')
    if model_name == 'xgboost':
        return xgb.XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric='logloss')
    if model_name == 'lightgbm':
        return lgb.LGBMClassifier(random_state=seed, class_weight='balanced')
    raise ValueError(f"Unknown model: {model_name}")

# -----------------------------
# Core functions
# -----------------------------

def run_cross_validation(training_data_path: str, k_features: int = 12, seed: int = 42) -> None:
    """
    Perform 10-fold CV to compare multiple models and print a summary table.
    """
    set_global_seed(int(seed))
    df_raw = pd.read_csv(training_data_path)
    df = engineer_features(df_raw)

    X = df.drop(columns=["BRS3", "patient_id"], errors='ignore')
    X = _ensure_numeric_df(X)
    y = df["BRS3"].astype(int)

    models_to_test = ['tabpfen', 'randomforest', 'xgboost', 'lightgbm']
    all_model_metrics = []

    for model_name in models_to_test:
        print(f"\n===== Testing Model: {model_name.upper()} =====")
        kf = KFold(n_splits=10, shuffle=True, random_state=int(seed))
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
            print(f"--- Fold {fold+1}/10 ---")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            n_feats = X_train.shape[1]
            k = max(1, min(k_features, n_feats)) if n_feats > 0 else 1
            fselect = SelectKBest(chi2, k=k)
            X_train_sel = fselect.fit_transform(X_train, y_train)
            selected_features = X_train.columns[fselect.get_support()]
            X_test_sel = X_test[selected_features]

            smote = SMOTE(random_state=int(seed))
            X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)

            clf = get_model(model_name, int(seed))
            clf.fit(X_train_res, y_train_res)

            probs = clf.predict_proba(X_test_sel)[:, 1]
            preds = clf.predict(X_test_sel)

            fold_results.append({
                'auroc': roc_auc_score(y_test, probs),
                'f1': f1_score(y_test, preds, pos_label=1),
                'precision': precision_score(y_test, preds, pos_label=1, zero_division=0),
                'recall': recall_score(y_test, preds, pos_label=1)
            })

        # Aggregate metrics for the current model
        avg_metrics = {metric: np.mean([res[metric] for res in fold_results]) for metric in fold_results[0]}
        all_model_metrics.append([model_name, avg_metrics['auroc'], avg_metrics['f1'], avg_metrics['precision'], avg_metrics['recall']])

    # Print comparison table
    headers = ["Model", "Mean AUROC", "Mean F1 (Class 1)", "Mean Precision (1)", "Mean Recall (1)"]
    print("\n\n" + "="*80)
    print(" " * 25 + "MODEL COMPARISON RESULTS")
    print("="*80)
    print(tabulate(all_model_metrics, headers=headers, floatfmt=".4f"))
    print("="*80)


def train_and_save_model(training_data_path: str, save_model_path: str, model_name: str, k_features: int = 12, seed: int = 42) -> None:
    """Train a specified model on the full dataset and save the bundle."""
    set_global_seed(int(seed))
    df_raw = pd.read_csv(training_data_path)
    df = engineer_features(df_raw)

    feature_columns = [c for c in df.columns if c not in ("BRS3", "patient_id")]
    X = _ensure_numeric_df(df[feature_columns])
    y = df["BRS3"].astype(int)

    n_feats = X.shape[1]
    k = max(1, min(k_features, n_feats)) if n_feats > 0 else 1
    fselect = SelectKBest(chi2, k=k)
    X_selected = fselect.fit_transform(X, y)

    smote = SMOTE(random_state=int(seed))
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)

    print(f"Training final model: {model_name.upper()}")
    clf = get_model(model_name, int(seed))
    clf.fit(X_resampled, y_resampled)

    bundle = {
        "model_name": model_name,
        "feature_columns": feature_columns,
        "selector": fselect,
        "clf": clf,
    }
    save_path = Path(save_model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_path)
    print(f"Model bundle saved to {save_path}")


def predict_with_saved_model(test_data_path: str, model_path: str, output_csv: str, seed: int = 42) -> None:
    """Load a saved bundle and run inference."""
    set_global_seed(int(seed))
    bundle: Dict[str, Any] = joblib.load(model_path)
    feature_columns: List[str] = bundle["feature_columns"]
    selector = bundle["selector"]
    clf = bundle["clf"]

    df_raw = pd.read_csv(test_data_path)
    df = engineer_features(df_raw)

    id_col = "patient_id" if "patient_id" in df.columns else "slide_id"
    X_raw = df.drop(columns=[c for c in ["BRS3", id_col] if c in df.columns], errors='ignore')
    X_aligned = _align_features(X_raw, feature_columns)
    X_sel = selector.transform(X_aligned)

    probs = clf.predict_proba(X_sel)
    preds = clf.predict(X_sel)
    labels = df["BRS3"].astype(int).tolist() if "BRS3" in df.columns else [-1] * len(df)

    results = pd.DataFrame({
        "slide_id": df[id_col].values,
        "label": labels,
        "probability": probs[:, 1],
        "prediction": preds,
    }).sort_values(by="slide_id")
    results.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison and prediction tool for CHIMERA")
    parser.add_argument("--mode", type=str, choices=["cv", "train", "predict"], required=True)
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--k_features", type=int, default=12, help="Number of features to select")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducibility")

    # Arguments for specific modes
    parser.add_argument("--model", type=str, choices=['tabpfen', 'randomforest', 'xgboost', 'lightgbm'], help="Model to use for train/predict modes")
    parser.add_argument("--save_model_path", type=str, default="model.joblib", help="Path to save trained model (train mode)")
    parser.add_argument("--model_path", type=str, default="model.joblib", help="Path to load model (predict mode)")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV path for predictions")

    args = parser.parse_args()

    if args.mode == "cv":
        run_cross_validation(args.input_file, k_features=args.k_features, seed=args.seed)
    elif args.mode == "train":
        if not args.model:
            raise SystemExit("--model is required for train mode")
        train_and_save_model(args.input_file, args.save_model_path, args.model, k_features=args.k_features, seed=args.seed)
    elif args.mode == "predict":
        if not args.model:
            raise SystemExit("--model is required for predict mode (to load the correct bundle)")
        predict_with_saved_model(args.input_file, args.model_path, args.output_csv, seed=args.seed)
