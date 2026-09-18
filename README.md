# Synthetic five-well manuscript reproduction

**Spyder users:** open `spyder_run_all.py` and run the whole file. See
[SPYDER_README.md](SPYDER_README.md) for setup and the other Spyder entry scripts.

Recreates all **19 manuscript figure types** with synthetic data (Figures 1–2 are
conceptual geological schematics), plus 14 supplementary plots. See
[FIGURE_GUIDE.md](FIGURE_GUIDE.md) for the figure-by-figure correspondence and
documented adaptations. Open `outputs/manuscript_figures/index.html` after a run.

Runnable Python code and generated results for synthetic carbonate well logs and
laboratory facies, porosity, and permeability. Includes comparison charts,
confusion matrices, correlation heatmaps, and depth tracks with computed scores.

## Use on GitHub

Upload this project's source files and the `.github/workflows/` and `tests/`
folders to your repository. If using `github-ready.zip`, extract it first and
upload its contents, including the hidden `.github` folder and `.gitignore`.
The ZIP itself is a download bundle, not the runnable repository structure.

The included GitHub Actions workflow runs the experiment and all tests on Linux
and Windows with Python 3.12 after pushes and pull requests. To run it manually,
open **Actions → Run five-well experiment → Run workflow** after the workflow is
on the default branch. Download the `five-well-results-...` artifact from a
completed run to obtain synthetic datasets, figures, metrics, and fitted models.
GitHub Actions must be enabled for the repository.

For interactive use in a GitHub Codespaces terminal:

```bash
python -m pip install -r requirements.txt
python run_experiment.py
python apply_models.py
python -m unittest discover -s tests -v
```

Generated outputs and local virtual environments are excluded by `.gitignore`;
the scripts regenerate everything. No account credentials or API keys are needed.
The workflow follows the [GitHub Python workflow documentation](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).

## Quick start

Python 3.11 or 3.12 is recommended. In this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_experiment.py
.\.venv\Scripts\python.exe apply_models.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Install Python first if the `python` command is unavailable. On macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_experiment.py
.venv/bin/python apply_models.py
.venv/bin/python -m unittest discover -s tests -v
```

Larger example (exact Gaussian processes are capped to control runtime):

```powershell
.\.venv\Scripts\python.exe run_experiment.py --samples 1400 --seed 42 --gp-max 500 --output outputs_large
```

Defaults: 360 depth samples per well, 1,800 rows overall, seed 42, and at most 240
training rows for each GP. All other models use all available training rows.
Increasing `--gp-max` above the development row count disables GP subsampling.

## Experiment design

- **Well_1, Well_2, Well_3:** development/testing through three leave-one-well-out
  folds. Each fold trains on two wells and tests on the remaining entire well.
- **Well_4, Well_5:** application wells. Selected models are fitted on the three
  development wells and applied to these two wells. Their synthetic laboratory
  labels are used only for the final evaluation and illustrations.
- The facies model is selected by mean development macro-F1; each regression
  model is selected by mean development R². Ties use the smaller model number.
- Imputation, feature standardization, and regression-target scaling are fitted
  within each training fold. No random split of neighboring depth samples is used.
- Predictors are **DT, GR, NPHI, RHOB, LogRT** only. Neither depth, well name,
  laboratory measurements, nor facies labels are used as predictors.
- Permeability is learned in log10 space and inverted to mD for prediction and
  metrics. Porosity is a fraction (0.20 = 20%), not a numeric percentage.
- All models are evaluated on the application wells for comparison, but that
  comparison is not used to select or tune models. Depth tracks show the eight
  best models ranked using development data only.

## Requested models and implementation

The list mixes classification and regression. **25 classifiers** predict facies;
**25 regression models** predict each continuous target. All 31 numbered entries
are implemented in their appropriate task. DA, NB, subspace DA, and RUSBoost do
not have direct regression equivalents; they are not applied to continuous labels.
LR means ordinary linear regression, not logistic regression, and is not used to
predict arbitrary facies integers.

| ID | Requested model | Settings / regression counterpart |
|---:|---|---|
| 1 | Fine DT | Up to 101 leaves (100 splits); tree regressor |
| 2 | Medium DT | Up to 21 leaves; tree regressor |
| 3 | Coarse DT | Up to 5 leaves; tree regressor |
| 4 | Linear DA | LDA with automatic shrinkage; classification only |
| 5 | Quadratic DA | QDA, regularization 0.05; classification only |
| 6 | Gaussian NB | Gaussian likelihood; classification only |
| 7 | Kernel NB | Independent Gaussian KDE per feature/class, bandwidth 0.3 after scaling; classification only |
| 8 | Linear SVM | Linear SVC / SVR, C=1 |
| 9 | Quadratic SVM | Polynomial degree 2, coef0=1; SVC / SVR |
| 10 | Cubic SVM | Polynomial degree 3, coef0=1; SVC / SVR |
| 11 | Fine Gaussian SVM | RBF scale sqrt(P)/4; SVC / SVR |
| 12 | Medium Gaussian SVM | RBF scale sqrt(P); SVC / SVR |
| 13 | Coarse Gaussian SVM | RBF scale 4 sqrt(P); SVC / SVR |
| 14 | Fine KNN | k=1; KNN regressor |
| 15 | Medium KNN | k=10; KNN regressor |
| 16 | Coarse KNN | k=100; KNN regressor |
| 17 | Cosine KNN | k=10, cosine distance; KNN regressor |
| 18 | Cubic KNN | k=10, Minkowski p=3; KNN regressor |
| 19 | Weighted KNN | k=10, inverse-square distance weights; KNN regressor |
| 20 | Boosted Tree EC | 40 SAMME AdaBoost trees, learning rate 0.1; 80 gradient-boosting regression trees, depth 3, rate 0.05 |
| 21 | Bagged Tree EC | 40 bootstrap decision trees; bagged regression trees |
| 22 | Subspace Discriminant EC | 30 LDA learners, 3 of 5 features, no row bootstrapping; classification only |
| 23 | Subspace KNN EC | 30 KNN learners, k=10, 3 of 5 features, no row bootstrapping; subspace KNN regression |
| 24 | RUS Boosted Tree EC | imbalanced-learn RUSBoost, 40 trees, 21 leaves, learning rate 0.1; classification only |
| 25 | GP | Gaussian-process classifier / regressor with fixed RBF kernel and training-row cap |
| 26 | LR | Ordinary least-squares linear regression; continuous targets only |
| 27 | Robust linear | Huber regression, epsilon 1.35 |
| 28 | Interaction linear | Pairwise interactions and main effects, ordinary least squares |
| 29 | Matérn GP | Fixed Matérn kernel, nu=1.5 |
| 30 | Rational-quadratic GP | Fixed rational-quadratic kernel, alpha=1 |
| 31 | Exponential GP | Fixed Matérn kernel, nu=0.5 |

`P=5`. The RBF implementation uses `gamma = 1 / (2 * scale**2)`. SVR uses epsilon
0.05 in standardized target space. GP uses RBF length scale sqrt(P), fixed signal
variance 1, and regression noise variance 0.05. No kernel optimization is performed.
GP sampling is stratified by training well and facies and never draws held-out rows.

These are **documented Python approximations** to the named MATLAB-style presets,
not exact MATLAB reproductions. Multiclass strategies, tree rules, regularization,
boosting variants, and hyperparameters can differ. No hyperparameter search or
nested validation is claimed.

## Files and figures

| File | Purpose |
|---|---|
| `synthetic_data.py` | Reproducible synthetic data with layered facies, well variation and measurement noise |
| `models.py` | Model registry, kernel NB, inverse-square KNN and preprocessing pipelines |
| `run_experiment.py` | Cross-well validation, model selection, fitting, scoring and figure generation |
| `apply_models.py` | Apply saved models to CSV logs without laboratory targets |
| `manuscript_figures.py` | All 19 numbered manuscript figure types and HTML gallery |
| `FIGURE_GUIDE.md` | Figure correspondence, model additions and scientific adaptations |
| `plots.py` | Correlation, confusion matrices, comparisons, point plots and depth tracks |
| `tests/test_workflow.py` | Mathematical and workflow checks |

After a run, `outputs/` contains:

- `data/`: separate log and laboratory CSVs for each of the five wells, development
  data, unlabeled application logs, and separate application evaluation truth.
- `metrics.csv`: accuracy, balanced accuracy and macro-F1 for facies; R², MSE, RMSE,
  and MAE in original target units for regression. Each model has one row per well.
- `predictions_all_models.csv`: out-of-fold development and held-out predictions.
- `cv_ranking.csv`, `model_registry.csv`, `selected_models.json` and `RESULTS.md`.
- `application_predictions.csv`: chosen-model predictions for both application wells.
- `models/`: three selected fitted pipelines, including transformations.
- `run_manifest.json` and `training_warnings.json`: settings, versions and fit warnings.
- `manuscript_figures/`: **19 numbered figures in PNG and PDF**, HTML gallery,
  figure manifest, and facies counts. Figures 1–2 are original conceptual schematics.
- `figures/`: **14 supplementary figures, each in PNG and vector PDF**:
  input logs; continuous-variable correlation heatmap; two-well confusion matrices
  with per-class TP/FN rates; three model comparisons; two laboratory/prediction
  point comparisons; and six depth-track figures (three targets × two wells).

For facies, tracks use accuracy, not R² on class codes. Pearson correlation excludes
nominal facies codes. Regression comparison figures use R², not an undefined
“regression accuracy.” All metrics are calculated; reference-image scores are never
hard-coded. Negative R² values are retained.

## Input schema and application to new logs

Required CSV columns: `Well, Depth_m, DT, GR, NPHI, RHOB, LogRT`.

| Column | Unit / convention |
|---|---|
| Depth_m | meters, increasing downward |
| DT | microseconds per foot |
| GR | API |
| NPHI | fraction |
| RHOB | g/cm³ |
| LogRT | log10 of RT in ohm·m |
| Porosity | laboratory fraction |
| Permeability | laboratory mD, strictly positive |
| Facies | 0=mudstone, 1=packstone/wackestone, 2=grainstone/packstone, 3=grainstone |

Log and laboratory rows are co-located at the same depths in this demonstration.
Real core measurements are usually sparser and require careful depth matching.
Use `LogRT = np.log10(RT_ohm_m)` for positive resistivity measurements. Missing
predictors can be NaN and are filled using fitted development medians.

```powershell
.\.venv\Scripts\python.exe apply_models.py --input my_well_logs.csv --models outputs/models --output my_predictions.csv
```

Only load trusted joblib files. The included trained models demonstrate the
workflow; they are not calibrated for real wells. Synthetic data use simplified
relations and do not establish geological validity or expected field performance.
To train on real data, replace the `generate(...)` call in `run_experiment.py` with
a validated, depth-aligned DataFrame with the same schema, and update the well
lists in `synthetic_data.py` before repeating the whole evaluation.

## Implementation references

- [MathWorks classifier options](https://www.mathworks.com/help/stats/choose-a-classifier.html)
- [MathWorks regression options](https://www.mathworks.com/help/stats/choose-regression-model-options.html)
- [scikit-learn bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
- [scikit-learn Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
- [imbalanced-learn RUSBoost](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.RUSBoostClassifier.html)
