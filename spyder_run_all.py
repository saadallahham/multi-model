"""Open this file in Spyder and run the WHOLE file (F5).

Keep all supplied Python modules in the same folder. Edit the settings below.
Results remain available in Spyder's Variable Explorer after execution.
"""
# %% User settings
SAMPLES_PER_WELL = 360       # At least 120
RANDOM_SEED = 42
GP_MAX_TRAINING_ROWS = 240   # At least 40; larger values increase runtime
OUTPUT_FOLDER = "outputs_spyder"  # Relative to this script, or an absolute path
OPEN_FIGURE_GALLERY = True

# %% Locate project independently of Spyder's working directory
from pathlib import Path
from types import SimpleNamespace
import sys
import webbrowser

PROJECT_FOLDER = Path(__file__).resolve().parent
if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))
output_directory = Path(OUTPUT_FOLDER).expanduser()
if not output_directory.is_absolute():
    output_directory = PROJECT_FOLDER / output_directory

# %% Generate five wells, train on Wells 1-3, test on Wells 4-5
import pandas as pd
from threadpoolctl import threadpool_limits
from run_experiment import run
from apply_models import apply

if SAMPLES_PER_WELL < 120 or GP_MAX_TRAINING_ROWS < 40:
    raise ValueError("Use SAMPLES_PER_WELL >= 120 and GP_MAX_TRAINING_ROWS >= 40.")
if RANDOM_SEED < 0:
    raise ValueError("RANDOM_SEED must be nonnegative.")

settings = SimpleNamespace(samples=SAMPLES_PER_WELL, seed=RANDOM_SEED,
                           gp_max=GP_MAX_TRAINING_ROWS, output=str(output_directory))
with threadpool_limits(limits=1):
    run(settings)
    apply(output_directory / "data/application_logs.csv",
          output_directory / "models", output_directory / "reapplied_predictions.csv")

# %% Inspect these DataFrames in Spyder's Variable Explorer
training_data = pd.read_csv(output_directory / "data/development.csv")
test_logs = pd.read_csv(output_directory / "data/application_logs.csv")
test_laboratory_data = pd.read_csv(output_directory / "data/application_truth_for_evaluation_only.csv")
test_predictions = pd.read_csv(output_directory / "application_predictions.csv")
metrics = pd.read_csv(output_directory / "metrics.csv")
model_ranking = pd.read_csv(output_directory / "cv_ranking.csv")
all_predictions = pd.read_csv(output_directory / "predictions_all_models.csv")

gallery_path = output_directory / "manuscript_figures/index.html"
print(f"\nFinished. Results: {output_directory}")
print(f"19 numbered figures (PNG/PDF): {gallery_path.parent}")
print("Training data, test data, predictions and metrics are ready in Variable Explorer.")
if OPEN_FIGURE_GALLERY:
    webbrowser.open(gallery_path.as_uri())
