"""Run saved models on a CSV of logs. Laboratory targets are not required."""
# %% Settings: relative to this script, or use absolute Windows paths
INPUT_CSV = "outputs_spyder/data/application_logs.csv"
MODEL_FOLDER = "outputs_spyder/models"
OUTPUT_CSV = "outputs_spyder/new_log_predictions.csv"
# Example: INPUT_CSV = r"C:\Users\YourName\Documents\my_logs.csv"

# %% Locate files and predict
from pathlib import Path
import sys

PROJECT_FOLDER = Path(__file__).resolve().parent
if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))

def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_FOLDER / path

import pandas as pd
from threadpoolctl import threadpool_limits
from apply_models import apply

input_path = project_path(INPUT_CSV)
model_directory = project_path(MODEL_FOLDER)
output_path = project_path(OUTPUT_CSV)
if not input_path.exists() or not (model_directory / "facies.joblib").exists():
    raise FileNotFoundError("Run spyder_run_all.py first, or correct INPUT_CSV and MODEL_FOLDER.")
with threadpool_limits(limits=1):
    apply(input_path, model_directory, output_path)
predictions = pd.read_csv(output_path)
print("Open 'predictions' in Spyder's Variable Explorer to inspect the results.")
