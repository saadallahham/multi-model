# Run the five-well experiment in Spyder

Extract **spyder-ready.zip** into one folder on your PC. Keep all Python files and
`requirements.txt` together. No GitHub account or manuscript file is needed.

## First run

1. Use a Python 3.11 or 3.12 environment with the packages in `requirements.txt`.
2. Open **spyder_run_all.py** in Spyder.
3. Edit the settings at the top if desired, then run the **whole file** with F5
   or the Run button. If prompted, choose execution in the current console to
   inspect the resulting variables.
4. The script generates data, fits models, evaluates the test wells, saves figures,
   and opens an HTML figure gallery in your browser.

If packages are missing, install them into the Python environment used by Spyder's
console. In that console, run the following once, replacing the example folder
with the folder where you extracted the ZIP:

```python
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r",
                       r"C:\path\to\spyder-ready\requirements.txt"])
```

Restart the Spyder kernel after installation, then run the script. This command
installs into the console's interpreter, not an unrelated system Python. An
environment managed by your institution may require its usual package setup.

## Scripts and settings

| Script | Use |
|---|---|
| `spyder_run_all.py` | Full experiment; start here |
| `spyder_redraw_figures.py` | Redraw the 19 figures from saved results without retraining |
| `spyder_predict_logs.py` | Apply saved models to another log CSV |

The remaining Python files contain the data generator, models, plotting functions
and shared workflow. They must remain beside the three Spyder scripts.

Default settings generate **360 samples per well**, with random seed 42 and a
240-row training cap for each Gaussian process. Change `SAMPLES_PER_WELL`,
`RANDOM_SEED`, `GP_MAX_TRAINING_ROWS` and `OUTPUT_FOLDER` near the top of the main
script. Use at least 120 samples per well and a GP cap of at least 40.

Outputs go to `outputs_spyder` beside the script regardless of Spyder's working
directory. An absolute `OUTPUT_FOLDER` is also accepted. Reusing that folder
overwrites generated results; use another folder to retain separate experiments.
No command-line arguments are required. Run the complete file first; its `# %%`
cells mark the stages for subsequent interactive inspection.

In Variable Explorer, inspect `training_data`, `test_logs`, `test_laboratory_data`,
`test_predictions`, `metrics`, `model_ranking` and `all_predictions`.

Plots are intentionally saved to files rather than opened as dozens of windows.
Browse `outputs_spyder/manuscript_figures/index.html`, or open the individual PNG
and PDF files. Set `OPEN_FIGURE_GALLERY = False` to suppress browser opening.
The 14 supplementary plots are in `outputs_spyder/figures`.

## Experiment design

Wells 1–3 are used for training, with leave-one-well-out development validation.
Models are selected using development scores and refitted on those three wells.
Wells 4–5 are held out for testing. Test laboratory data are used for evaluation
only. The scripts preserve the existing synthetic-data generator and models.

All 19 manuscript figure types are generated. Figures 1–2 are conceptual
schematics; numerical results come from synthetic data and will differ from the
manuscript. See `FIGURE_GUIDE.md` for the detailed correspondence.

For new logs, set `INPUT_CSV` in `spyder_predict_logs.py`. Required columns are
`Well, Depth_m, DT, GR, NPHI, RHOB, LogRT`; units are documented in `README.md`.
The supplied models are trained on synthetic data and are for demonstration.
Only load model files you trust.
