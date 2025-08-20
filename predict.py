"""
Grand Challenge-style inference entrypoint (Task 2, clinical-only TabPFN).

- Reads clinical JSON from CHIMERA_CLINICAL_JSON (default: /input/chimera-clinical-data-of-bladder-cancer-patients.json)
- Loads a pre-trained TabPFN model bundle from CHIMERA_MODEL (default: /model/model.joblib)
- Writes a single float probability to CHIMERA_OUTPUT_JSON (default: /output/brs-probability.json)

The model bundle must be created beforehand using:
  python -m tabpfn_final --mode train --input_file <train.csv> --save_model_path /model/model.joblib

Notes:
- We ignore WSI and tissue mask inputs as this submission uses clinical data only.
- Missing clinical fields are imputed with 0; negatives are clipped to 0 to satisfy chi2.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd


def _ensure_numeric_row(row: pd.DataFrame) -> pd.DataFrame:
    # Convert to numeric, coerce errors to NaN, fill NaN with 0, and clip negatives to 0
    row = row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row[row < 0] = 0.0
    return row


def main() -> int:
    clinical_json_path = os.environ.get(
        "CHIMERA_CLINICAL_JSON", "/input/chimera-clinical-data-of-bladder-cancer-patients.json"
    )
    output_json_path = os.environ.get("CHIMERA_OUTPUT_JSON", "/output/brs-probability.json")
    model_path = os.environ.get("CHIMERA_MODEL", "/model/model.joblib")

    in_p = Path(clinical_json_path)
    out_p = Path(output_json_path)
    model_p = Path(model_path)

    if not in_p.is_file():
        print(f"[ERROR] Clinical JSON not found: {in_p}")
        return 2
    if not model_p.is_file():
        print(f"[ERROR] Model file not found: {model_p}")
        return 3

    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Determine seed for reproducibility (not strictly needed for inference, but kept for determinism)
    seed_env = os.environ.get("CHIMERA_SEED", "42")
    try:
        seed = int(seed_env)
    except Exception:
        seed = 42

    try:
        bundle: Dict[str, Any] = joblib.load(model_p)
        feature_columns = bundle["feature_columns"]
        selector = bundle["selector"]
        clf = bundle["clf"]
    except Exception as e:
        print(f"[ERROR] Failed to load model bundle: {e}")
        return 4

    # Read clinical JSON
    try:
        with open(in_p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read clinical JSON: {e}")
        return 5

    # Support either a dict of fields or a list with one dict
    if isinstance(data, list):
        if len(data) == 0:
            print("[ERROR] Clinical JSON list is empty.")
            return 6
        sample = data[0]
    elif isinstance(data, dict):
        sample = data
    else:
        print("[ERROR] Unexpected JSON format. Must be an object or a single-element list.")
        return 6

    # Build single-row DataFrame with expected feature columns
    row_dict = {}
    for col in feature_columns:
        val = sample.get(col, 0)
        row_dict[col] = val
    X = pd.DataFrame([row_dict], columns=feature_columns)
    X = _ensure_numeric_row(X)

    try:
        # Transform with saved selector and predict probability for class 1 (BRS3)
        X_sel = selector.transform(X)
        prob = float(clf.predict_proba(X_sel)[0, 1])
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return 7

    # Write a single float to JSON file
    try:
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(prob, f)
            f.write("\n")
    except Exception as e:
        print(f"[ERROR] Failed to write output JSON: {e}")
        return 8

    print(f"[INFO] Wrote BRS probability to {out_p}: {prob:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
