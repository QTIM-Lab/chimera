# CHIMERA — Entrypoint, Structure, and Usage

This repository contains the challenge baseline and your code to run a simple prediction workflow with TabPFN on clinical data (Task 2). The entrypoint, project structure, and usage are documented below.

## Entrypoint
- `app.py`: Recommended command-line interface (CLI) entry point to run the TabPFN workflow without modifying the baseline.
- `scripts\run_tabpfn.ps1`: Helper script (PowerShell/Windows) that provides a convenient way to call `app.py`.
- `tabpfn_final.py`: Original implementation of the training and prediction flow with KFold cross-validation using TabPFN.

You can invoke the workflow in two ways:
1) Directly using Python:
   ```shell
   python .\app.py --input_file <path_to_your.csv> --output_csv <output.csv>
   ```

2) Using the PowerShell script (Windows):
   ```shell
   .\scripts\run_tabpfn.ps1 -InputFile <path_to_your.csv> -OutputCsv <output.csv>
   ```

## Project Structure (Top Level)
- `app.py`                 → Recommended CLI Entry point
- `tabpfn_final.py`        → KFold + TabPFN logic for Task 2
- `requirements.txt`       → Project dependencies
- `scripts\run_tabpfn.ps1` → PowerShell wrapper for Windows
- `baseline\`              → Baselines and utilities provided by the challenge (unmodified)
- Other utilities          → e.g., `pred_metrics_calculator.py`, `integrator_mean.py`

Note: All preparation for Docker/submission will be done at the project's top level (without modifying the `baseline\` directory).

## Installation
1) Create and activate a Python environment (recommended):
   ```shell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2) Install dependencies:
   ```shell
   pip install -r requirements.txt
   ```

Main requirements: `tabpfn`, `scikit-learn`, `pandas`, `numpy`.

## Input Format (Clinical CSV)
The input CSV must contain at least the following columns:
- `patient_id`: Identifier for the patient/sample.
- `BRS3`: The target binary label (1 for BRS3, 0 for BRS1/2).
- Other numeric columns: Clinical variables to be used as features.

Default sample CSV: .\dataset\clinical_data_BRS_binary.csv

Considerations:
- The pipeline performs feature selection using the chi-squared (chi2) test and replaces negative values with 0 before this step.
- KFold cross-validation is executed (10 folds by default), and the predictions for each sample are saved.

## Output
A CSV file is generated with the following columns:
- `slide_id`: Corresponds to the `patient_id` from the original row.
- `label`: The actual label (BRS3, 0 or 1).
- `probability`: The predicted probability of the positive class (BRS3=1).
- `prediction`: The predicted class (0 or 1).

The file is saved with the name specified by the `--output_csv` argument (default: `tabpfn_pred_probs.csv`), and the rows are sorted by `slide_id`.

## Quick Usage
- With Python:
  ```shell
  python .\app.py --input_file .\dataset\clinical_data_BRS_binary.csv --output_csv .\outputs\tabpfn_pred_probs.csv
  ```

- With PowerShell (Windows):
  ```shell
  .\scripts\run_tabpfn.ps1 -InputFile .\dataset\clinical_data_BRS_binary.csv -OutputCsv .\outputs\tabpfn_pred_probs.csv
  ```

To view the help message:

## Model Training and Saved Bundle (for Grand Challenge Inference)
You need to train once on labeled clinical data and save a model bundle (feature selector + TabPFN + feature schema). This produces a single file you will mount into the inference container.

- Train and save the bundle (Windows PowerShell example):
  ```shell
  python -m tabpfn_final --mode train --input_file .\dataset\clinical_data_BRS_binary.csv --save_model_path .\model\model.joblib --k_features 12
  ```

- To run cross-validated evaluation (optional):
  ```shell
  python -m tabpfn_final --mode cv --input_file .\dataset\clinical_data_BRS_binary.csv --output_csv .\outputs\tabpfn_cv_preds.csv --k_features 12
  ```

- To run local prediction with a saved model (no labels required in the CSV):
  ```shell
  python -m tabpfn_final --mode predict --input_file .\dataset\clinical_data_BRS_binary.csv --model_path .\model\model.joblib --output_csv .\outputs\predictions.csv
  ```

Notes:
- The training CSV must contain columns: patient_id, BRS3, and numeric features.
- NaNs are filled with 0 and negatives clipped to 0 to satisfy chi2; ensure your features are numeric.

## Docker Inference Container (Grand Challenge Style)
This repository includes a container entrypoint and Dockerfile to run inference in a Grand Challenge environment.

- Build the image:
  ```shell
  docker build -t chimera-tabpfn:latest .
  ```

- Run inference by mounting your clinical JSON, the saved model, and an output directory:
  ```shell
  docker run --rm \
    -e CHIMERA_CLINICAL_JSON=/input/chimera-clinical-data-of-bladder-cancer-patients.json \
    -e CHIMERA_OUTPUT_JSON=/output/brs-probability.json \
    -e CHIMERA_MODEL=/model/model.joblib \
    -v C:\\path\\to\\patient.json:/input/chimera-clinical-data-of-bladder-cancer-patients.json:ro \
    -v C:\\dataset\\out:/output \
    -v C:\\path\\to\\model.joblib:/model/model.joblib:ro \
    chimera-tabpfn:latest
  ```

- Default environment variables inside the container:
  - CHIMERA_CLINICAL_JSON=/input/chimera-clinical-data-of-bladder-cancer-patients.json
  - CHIMERA_OUTPUT_JSON=/output/brs-probability.json
  - CHIMERA_MODEL=/model/model.joblib

The resulting /output/brs-probability.json will contain a single float value: the probability of BRS3 (BCG response subtype).


## Reproducibility
You can make runs deterministic by setting a global seed.

- Python CLI (CV):
  ```shell
  python .\app.py --input_file .\dataset\clinical_data_BRS_binary.csv --output_csv .\outputs\tabpfn_pred_probs.csv --seed 42
  ```
- Module CLI:
  ```shell
  python -m tabpfn_final --mode cv --input_file .\dataset\clinical_data_BRS_binary.csv --output_csv .\outputs\tabpfn_cv_preds.csv --k_features 12 --seed 42
  python -m tabpfn_final --mode train --input_file .\dataset\clinical_data_BRS_binary.csv --save_model_path .\model\model.joblib --k_features 12 --seed 42
  python -m tabpfn_final --mode predict --input_file .\dataset\clinical_data_BRS_binary.csv --model_path .\model\model.joblib --output_csv .\outputs\predictions.csv --seed 42
  ```
- Docker inference: set CHIMERA_SEED (defaults to 42 if not set)
  ```shell
  docker run --rm \
    -e CHIMERA_INPUT=/input/clinical.csv \
    -e CHIMERA_OUTPUT=/output/predictions.csv \
    -e CHIMERA_MODEL=/model/model.joblib \
    -e CHIMERA_SEED=42 \
    -v C:\\data\\clinical_test.csv:/input/clinical.csv:ro \
    -v C:\\data\\out:/output \
    -v C:\\path\\to\\model.joblib:/model/model.joblib:ro \
    chimera-tabpfn:latest
  ```

Notes:
- Seeds are applied to Python, NumPy, and PyTorch (CPU). Some PyTorch operations may remain nondeterministic on certain backends; we attempt to enforce deterministic algorithms when possible.
