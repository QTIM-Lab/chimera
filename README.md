# CHIMERA Task 2 — Grand Challenge Docker Submission Guide

This repository provides a ready-to-run Docker container for Grand Challenge inference using a TabPFN model on clinical data only. Follow the steps below to prepare the model bundle, build the image, test locally, and submit.

What you will submit
- A Docker image whose entrypoint reads a clinical JSON file and writes a single float (probability of BRS3) to an output JSON file.
- The container loads a pre-trained model bundle at runtime (recommended) or you can bake the model into the image at build time.

Container I/O (defaults inside the container)
- Input clinical JSON: CHIMERA_CLINICAL_JSON=/input/chimera-clinical-data-of-bladder-cancer-patients.json
- Output probability JSON: CHIMERA_OUTPUT_JSON=/output/brs-probability.json
- Model bundle (joblib): CHIMERA_MODEL=/model/model.joblib

Prerequisites
- Docker installed on your machine.
- Python 3.10 or 3.11 to train and save the model bundle (one-time).
- Architecture: linux/amd64 (no ARM builds).

1) Train once and save the model bundle (required)
Use the provided training utility to create a serialized bundle that includes the feature selector, the TabPFN classifier, and the feature schema.

Linux (bash) example:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m tabpfn_final --mode train --input_file ./dataset/clinical_data_BRS_binary.csv --save_model_path ./model/model.joblib --k_features 12 --seed 42
```
Notes
- The training CSV must contain: patient_id, BRS3, and numeric feature columns.
- Missing values are filled with 0 and negatives are clipped to 0 to satisfy chi2-based feature selection.
- The output ./model/model.joblib will be used for inference.

2) Build the Docker image
Option A — Model mounted at runtime (recommended)
```bash
docker build -t chimera-tabpfn:latest .
```

Option B — Bake the model into the image
If model/model.joblib exists before building, the Dockerfile will copy it into /model/model.joblib inside the image automatically.
```bash
# Ensure ./model/model.joblib exists
docker build -t chimera-tabpfn:with-model .
```

3) Test the container locally
Option A — Mount the model at runtime
```bash
# Replace the host paths with your actual locations
# Linux example with volume mounts

docker run --rm \
  -e CHIMERA_CLINICAL_JSON=/input/chimera-clinical-data-of-bladder-cancer-patients.json \
  -e CHIMERA_OUTPUT_JSON=/output/brs-probability.json \
  -e CHIMERA_MODEL=/model/model.joblib \
  -v /path/to/patient.json:/input/chimera-clinical-data-of-bladder-cancer-patients.json:ro \
  -v /path/to/outdir:/output \
  -v /path/to/model.joblib:/model/model.joblib:ro \
  chimera-tabpfn:latest
```
This writes a single float to /output/brs-probability.json.

Option B — Image with embedded model
```bash
# Requires you built chimera-tabpfn:with-model (step 2B)
docker run --rm \
  -e CHIMERA_CLINICAL_JSON=/input/chimera-clinical-data-of-bladder-cancer-patients.json \
  -e CHIMERA_OUTPUT_JSON=/output/brs-probability.json \
  chimera-tabpfn:with-model
```

4) Export an image file for submission (optional)
If you need to upload a tarball of the image:
```bash
docker save -o chimera-tabpfn_latest.tar chimera-tabpfn:latest
```

FAQ and troubleshooting
- Missing model file error: ensure you either mount a valid model.joblib at /model/model.joblib or build the image with the model included (step 2B).
- Clinical JSON structure: the container expects a JSON object with keys matching the training feature names. Missing keys are treated as 0; values are coerced to numeric and negatives are clipped to 0.
- Determinism: you can set CHIMERA_SEED to control RNG seeds, though inference is deterministic by design.

Reference files in this repo
- predict.py: container entrypoint that performs inference and writes the probability JSON.
- Dockerfile: builds the runtime image and sets defaults for CHIMERA_* variables.
- tabpfn_final.py: utilities to train the model bundle used by the container.

That’s it. After verifying locally, package or push your image per the challenge’s submission instructions.
