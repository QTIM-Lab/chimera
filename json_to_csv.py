"""
json_to_csv.py
Dagoberto Pulido-Arias, Aylin Ergun
Converts a directory of clinical data JSON files into a single CSV file.

This script iterates through all files in a specified input directory,
reads each file ending with '.json', and compiles them into a single
pandas DataFrame. The patient ID, derived from the filename, is added
as a new column. The final DataFrame is then saved to a specified
CSV file.
"""

import os
import json
import pandas as pd
import argparse


def convert_json_to_csv(json_dir, csv_path):
    """
    Reads all JSON files from a directory and saves them as a single CSV.

    Parameters
    ----------
    json_dir : str
        The path to the directory containing the JSON files.
    csv_path : str
        The path where the output CSV file will be saved.

    Returns
    -------
    None
    """
    if not os.path.isdir(json_dir):
        print(f"Error: Directory not found at {json_dir}")
        return

    data_list = []
    print(f"Reading JSON files from: {json_dir}")

    for subdir in os.listdir(json_dir):
        subdir = os.path.join(json_dir, subdir)
        if os.path.isdir(subdir):
            for filename in os.listdir(subdir):
                if filename.endswith('.json'):
                    file_path = os.path.join(subdir, filename)
                    with open(file_path, 'r') as f:
                        try:
                            data = json.load(f)
                            data = __nominal_data_handler__(filename, data)
                            data_list.append(data)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from {filename}. Skipping.")

    if not data_list:
        print("No valid JSON files found. Exiting.")
        return

    df = pd.DataFrame(data_list)
    
    # Ensure all columns from the first file are present for all records
    # This handles cases where some JSONs might have missing keys
    # df = df.reindex(columns=sorted(df.columns))

    df.to_csv(csv_path, index=False)
    print(f"Successfully converted {len(data_list)} JSON files to {csv_path}")


def __nominal_data_handler__(filename, data):
    """
    Encodes nominal clinical data into one-hot format

    Parameters
    ----------
    filename : str
        Filename to get patient ID from
    data : dict
        Raw data from the json 

    Returns
    -------
    data : dict
        Updated data
    """
    # Patient ID
    data['patient_id'] = filename.replace('_CD.json', '')

    # Sex
    if data['sex'] == "Female":
        data['female'] = True
        data['male'] = False
    else:
        data['female'] = False
        data['male'] = True
    del data['sex']

    # Smoking status
    if data['smoking'] == "Yes":
        data['smoker'] = True
        data['non-smoker'] = False
    else:
        data['smoker'] = False
        data['non-smoker'] = True
    del data['smoking']

    # Tumor recurrence type
    if data['tumor'] == "Primary":
        data['primary_tumor'] = True
        data['recurrent_tumor'] = False
    else:
        data['primary_tumor'] = False
        data['recurrent_tumor'] = True
    del data['tumor']

    # Stage 
    if data['stage'] == "TaHG":
        data['tahg'] = True
        data['t1hg'] = False
        data['t2hg'] = False
    elif data['substage'] == "T1HG":
        data['tahg'] = False
        data['t1hg'] = True
        data['t2hg'] = False
    else:
        data['tahg'] = False
        data['t1hg'] = False
        data['t2hg'] = True
    del data['stage']

    # Substage 
    if data['substage'] == "T1e":
        data['t1e'] = True
        data['t1m'] = False
        data['substage_na'] = False
    elif data['substage'] == "T1m":
        data['t1e'] = False
        data['t1m'] = True
        data['substage_na'] = False
    else:
        data['t1e'] = False
        data['t1m'] = False
        data['substage_na'] = True
    del data['substage']

    # Grade
    if data['grade'] == "G3":
        data['g3'] = True
        data['g2'] = False
    else:
        data['g3'] = False
        data['g2'] = True
    del data['grade']

    # reTUR administered or not
    if data['reTUR'] == "Yes":
        data['retur'] = True
        data['no_retur'] = False
    else:
        data['retur'] = False
        data['no_retur'] = True
    del data['reTUR']

    # Lymphovascular invasion status
    if data['LVI'] == "Yes":
        data['lvi_observed'] = True
        data['no_lvi_observed'] = False
    else:
        data['lvi_observed'] = False
        data['no_lvi_observed'] = True
    del data['LVI']

    # Cancer histology
    if data['variant'] == "UCC":
        data['ucc_only'] = True
        data['ucc_variant'] = False
    else:
        data['ucc_only'] = False
        data['ucc_variant'] = True
    del data['variant']

    # EORTC classification
    if data['EORTC'] == "Highest risk":
        data['highest_risk'] = True
        data['high_risk'] = False
    else:
        data['highest_risk'] = False
        data['high_risk'] = True
    del data['EORTC']

    # BRS classification from molecular marker -- one hot
    if data['BRS'] == "BRS1":
        data['BRS1'] = True
        data['BRS2'] = False
        data['BRS3'] = False
    elif data['BRS'] == "BRS2":
        data['BRS1'] = False
        data['BRS2'] = True
        data['BRS3'] = False
    else:
        data['BRS1'] = False
        data['BRS2'] = False
        data['BRS3'] = True
    del data['BRS']

    # BRS classification from molecular marker -- binary BRS3 vs 1/2
    if data['BRS'] == "BRS3":
        data['BRS3'] = True
    else:
        data['BRS3'] = False
    del data['BRS']

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a folder of JSON files to a single CSV."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory with JSON files."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="clinical_data.csv",
        help="Path for the output CSV file."
    )
    args = parser.parse_args()

    convert_json_to_csv(args.input_dir, args.output_csv)