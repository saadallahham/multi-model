# Well logs and laboratory data

This repository provides a reproducible Python example for predicting carbonate
facies, porosity and permeability from well logs. It generates five synthetic
wells, develops models using Wells 1–3, and evaluates them on held-out Wells 4–5.
You can run the project in Spyder or from a terminal.

The repository includes generated CSV data, fitted example models, evaluation
results, 19 numbered manuscript figure types and 14 supplementary plots.
**All data are synthetic.** Figures 1–2 are conceptual geological schematics;
the other figures are computed from synthetic data and model predictions.
They do not reproduce the manuscript's original measurements or numerical scores.

## 1. Download the project

On this repository's main page, select **Code → Download ZIP**, then extract the
ZIP to a folder on your computer. Keep all Python files together. Do not run the
scripts from inside the ZIP archive.

Alternatively, clone the repository:

```bash
git clone https://github.com/saadallahham/synthetic-well-logs.git
cd synthetic-well-logs
```

No manuscript file, API key or external dataset is needed.

## 2. Explore the included data and figures

You can inspect the saved results before installing Python:

- [Figure guide](FIGURE_GUIDE.md): correspondence to manuscript figures and documented adaptations.


## 3. Install the requirements

Use **Python 3.11 or 3.12**. The pinned machine-learning dependencies are listed
in `requirements.txt`. Install them in the same environment that runs the code.

### If you use Spyder

Open Spyder's IPython console and run the following once. Replace the example
path with the location of your extracted project:

```python
import sys
import subprocess
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    r"C:\path\to\synthetic-well-logs-main\requirements.txt"
])
```

Restart the Spyder kernel after installing the packages. Spyder itself is a
separate application and is not installed by `requirements.txt`.

### If you use a terminal

Open a terminal in the extracted project folder. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_experiment.py
```

On macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_experiment.py
```

The terminal workflow writes to `outputs/` by default. The Spyder workflow below
writes to `outputs_spyder/`. These folders have the same result structure.

## 4. Run the complete experiment in Spyder

1. Open **spyder_run_all.py** in Spyder.
2. Edit the settings at the top if necessary.
3. Run the **whole file** using F5 or the Run button.
4. Inspect the printed progress and the generated browser gallery.
5. Open the DataFrames in Variable Explorer to inspect data and results.

The main script generates data, validates models within the three training wells,
selects models, fits them on those three wells, evaluates the two test wells,
saves predictions and creates all figures. It finds its files relative to the
script location, regardless of Spyder's current working directory.

| Setting | Default | Meaning |
|---|---:|---|
| `SAMPLES_PER_WELL` | 360 | Depth samples per well; minimum 120 |
| `RANDOM_SEED` | 42 | Nonnegative seed for reproducibility |
| `GP_MAX_TRAINING_ROWS` | 240 | Training-row cap for Gaussian processes; minimum 40 |
| `OUTPUT_FOLDER` | `outputs_spyder` | Relative to the script, or an absolute folder path |
| `OPEN_FIGURE_GALLERY` | `True` | Open the HTML gallery when the run finishes |

With the default settings, the generator creates **1,800 rows across five wells**.
Increasing the Gaussian-process cap can substantially increase runtime. Running
again in the same output folder replaces generated results; choose a new
`OUTPUT_FOLDER` to retain separate runs.

Available variables include `training_data`, `test_logs`, `test_laboratory_data`,
`test_predictions`, `metrics`, `model_ranking` and `all_predictions`.

## 5. Redraw figures without retraining

In Spyder, open **spyder_redraw_figures.py**, set `OUTPUT_FOLDER` to the existing
results folder, and run the whole file. This redraws the 19 numbered figures.

From a terminal with the dependencies installed:

```bash
python manuscript_figures.py --output outputs_spyder
```

The PNG and PDF files are saved to `manuscript_figures/` within that output folder.

## 6. Apply fitted models to another log CSV

In Spyder, open **spyder_predict_logs.py** and edit `INPUT_CSV`, `MODEL_FOLDER`
and `OUTPUT_CSV`. Run the whole file. The resulting `predictions` DataFrame is
available in Variable Explorer.

From a terminal:

```bash
python apply_models.py --input my_logs.csv --models outputs_spyder/models --output my_predictions.csv
```

The input CSV must contain the following columns:

| Column | Description and units |
|---|---|
| `Well` | Well identifier |
| `Depth_m` | Depth in meters; each well/depth pair must be unique |
| `DT` | Sonic transit time, microseconds per foot |
| `GR` | Gamma ray, API |
| `NPHI` | Neutron porosity as a fraction |
| `RHOB` | Bulk density, g/cm³ |
| `LogRT` | Base-10 logarithm of resistivity in ohm·m |

For positive resistivity values, calculate `LogRT = np.log10(RT_ohm_m)`.
Missing predictors may be NaN and are filled using the fitted training medians.
Laboratory targets are not required for prediction.

The laboratory CSVs contain `Facies`, `Porosity` (fraction), and `Permeability`
(mD). Facies codes are 0 = mudstone, 1 = packstone/wackestone,
2 = grainstone/packstone, and 3 = grainstone.

The included fitted models demonstrate the synthetic workflow; they are not
calibrated for field applications. Only load trusted joblib model files.

## 7. Understand the training and test split

- **Wells 1–3:** development data. Leave-one-well-out validation trains on two
  wells and validates on the third, repeating for all three wells.
- **Wells 4–5:** held-out test data. Their labels are used only for final
  evaluation and illustrations, not for model selection or preprocessing.
- Classifiers are selected by mean development macro-F1; regressors by mean
  development R². Selected models are refitted on all three training wells.
- Inputs are DT, GR, NPHI, RHOB and LogRT. Depth, well identity, laboratory
  measurements and facies labels are not predictors.
- Preprocessing is fitted inside each training fold. Permeability is learned in
  log10 space; its evaluation metrics are calculated in mD.

The registry includes **25 classifiers** and **25 regressors for each continuous
target**. MATLAB-style model names are implemented as documented Python
approximations. See [FIGURE_GUIDE.md](FIGURE_GUIDE.md) and
[model_registry.csv](outputs_spyder/model_registry.csv) for details.

Facies scores use accuracy, balanced accuracy and macro-F1. Continuous targets
use R², MSE, RMSE and MAE. Negative R² values are retained. Geological realism and
performance on real wells cannot be established from this synthetic experiment.

## 8. Files and outputs

| File or folder | Purpose |
|---|---|
| `spyder_run_all.py` | Main entry point for Spyder users |
| `spyder_redraw_figures.py` | Regenerate numbered figures |
| `spyder_predict_logs.py` | Predict from another log CSV |
| `synthetic_data.py` | Generate reproducible synthetic logs and laboratory data |
| `models.py` | Model definitions and preprocessing pipelines |
| `run_experiment.py` | Training, validation, selection and evaluation |
| `plots.py` | Supplementary figures |
| `manuscript_figures.py` | Numbered manuscript figures and gallery |
| `apply_models.py` | Command-line prediction from saved models |
| `tests/test_workflow.py` | Reproducibility and result verification checks |
| `outputs_spyder/` | Included demonstration data, figures, models and results |

Inside the output folder, `data/development.csv` combines Wells 1–3.
`data/application_logs.csv` contains Wells 4–5, and
`data/application_truth_for_evaluation_only.csv` stores their evaluation labels.
`application_predictions.csv` contains selected-model predictions.
`predictions_all_models.csv` contains all model predictions; `cv_ranking.csv`
records the development ranking. Fitted models are stored in `models/`.

## 9. Verify a complete run

From the project folder, using the environment with the installed dependencies:

```bash
python run_experiment.py
python apply_models.py
python -m unittest discover -s tests -v
```

The integration checks read the default `outputs/` folder, so run these commands
in order. The GitHub Actions workflow performs the same process on Linux and
Windows. Repository owners and contributors with sufficient permissions can
also use **Actions → Run five-well experiment → Run workflow**. Completed runs
provide downloadable result artifacts when the workflow succeeds.

## Troubleshooting

- **ModuleNotFoundError:** install `requirements.txt` in the interpreter used by
  Spyder's console, then restart the kernel.
- **Missing saved results:** run `spyder_run_all.py` first or correct the output
  folder setting. The downloaded `outputs_spyder/` folder also contains an example run.
- **No plots in Spyder's Plots pane:** the scripts save figures to PNG/PDF and
  display an HTML gallery instead of opening many plot windows.
- **Different output location:** Spyder defaults to `outputs_spyder/`; the
  command-line experiment defaults to `outputs/`.
- **Using real laboratory data:** this requires adapting the training workflow
  and depth matching. `apply_models.py` predicts only; it does not retrain models.
