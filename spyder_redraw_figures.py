"""Run the whole file in Spyder to redraw the 19 figures without retraining."""
# %% Settings: must match the output folder used in spyder_run_all.py
OUTPUT_FOLDER = "outputs_spyder"
OPEN_FIGURE_GALLERY = True

# %% Load the saved experiment
from pathlib import Path
import sys
import webbrowser

PROJECT_FOLDER = Path(__file__).resolve().parent
if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))
output_directory = Path(OUTPUT_FOLDER).expanduser()
if not output_directory.is_absolute():
    output_directory = PROJECT_FOLDER / output_directory
if not (output_directory / "cv_ranking.csv").exists():
    raise FileNotFoundError("Run spyder_run_all.py first, or correct OUTPUT_FOLDER.")

import pandas as pd
from manuscript_figures import render_all

training_data = pd.read_csv(output_directory / "data/development.csv")
test_logs = pd.read_csv(output_directory / "data/application_logs.csv")
test_laboratory_data = pd.read_csv(output_directory / "data/application_truth_for_evaluation_only.csv")
data = pd.concat([training_data, test_logs.merge(test_laboratory_data,
                  on=["Well", "Depth_m"], validate="one_to_one")], ignore_index=True)
all_predictions = pd.read_csv(output_directory / "predictions_all_models.csv")
metrics = pd.read_csv(output_directory / "metrics.csv")
model_ranking = pd.read_csv(output_directory / "cv_ranking.csv")

# %% Regenerate PNG/PDF and browser gallery
render_all(data, all_predictions, metrics, model_ranking, output_directory / "manuscript_figures")
gallery_path = output_directory / "manuscript_figures/index.html"
print(f"Figures regenerated: {gallery_path.parent}")
if OPEN_FIGURE_GALLERY:
    webbrowser.open(gallery_path.as_uri())
