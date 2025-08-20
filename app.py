"""
app.py
Author: Repo entry point wrapper

Purpose:
- Provide a stable, simple CLI entry point for running TabPFN over CHIMERA clinical dataset.
- Keep original code in tabpfn_final.py untouched while enabling Docker-friendly invocation.

Usage:
  python app.py --input_file <path> --output_csv <file>
"""
from pathlib import Path
import argparse
import sys

# Import the original implementation
try:
    from tabpfn_final import tabpfn_predict
except Exception as e:
    print("Error: Unable to import tabpfn_final.tabpfn_predict. Ensure the file exists and dependencies are installed.")
    print(f"Details: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run TabPFN to make a KFold cross validated prediction on the given dataset."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input dataset file. Will be split into training and test sets automatically.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="tabpfn_pred_probs.csv",
        help="Name for the output CSV file containing scores and predictions per sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for reproducibility (affects KFold and internal RNGs)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"Error: File not found at {input_path}")
        sys.exit(2)

    tabpfn_predict(str(input_path), args.output_csv, seed=args.seed)


if __name__ == "__main__":
    main()
