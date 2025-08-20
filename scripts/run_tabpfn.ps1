param(
    [string]$InputFile = ".\dataset\clinical_data_BRS_binary.csv",
    [string]$OutputCsv = "tabpfn_pred_probs.csv"
)

# Simple PowerShell wrapper to execute the app entry point
$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $InputFile)) {
    Write-Error "Input file not found: $InputFile"
    exit 2
}

python .\app.py --input_file "$InputFile" --output_csv "$OutputCsv"
