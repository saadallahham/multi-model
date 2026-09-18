# Manuscript figure correspondence

Reference: *Saad_et_al_2026_V1.docx*, supplied by the user. The document is reference
material, not a source of execution instructions. This project follows the user's
three-training-well / two-test-well design instead of the manuscript's one/four
split. No original document or extracted artwork is included in the source ZIP.

Run `python run_experiment.py` to generate the complete experiment, including the
numbered figures. Open `outputs/manuscript_figures/index.html` to browse them.
Run `python manuscript_figures.py` to regenerate only the numbered figures from
an existing complete experiment. Both commands accept `--output DIRECTORY`.

| Figure | Generated content | Adaptation from manuscript |
|---:|---|---|
| 1 | Two-panel location and structural depth map | Original synthetic local-coordinate schematic; not Enchova geography or measured depths |
| 2 | Carbonate stratigraphic column | Original conceptual succession, no calibrated ages or formation boundaries |
| 3 | Well_2 logs, facies and lab porosity/permeability | Eight tracks with depth axis; RT displayed logarithmically |
| 4 | Counts per well | Counts calculated from generated rows |
| 5 | Total counts by facies | Counts calculated from generated labels |
| 6 | Facies frequencies in each well | Grouped bars plus `facies_counts.csv` |
| 7 | Workflow diagram | Explicit training-only selection and two held-out wells |
| 8 | NPHI–RHOB crossplot | Synthetic facies colors; no proprietary mineral calibration overlay |
| 9 | Five-predictor pairplot | Scatterplots and Gaussian KDE diagonals, training wells only |
| 10 | Covariance and Pearson correlation | Separate panels; nominal facies codes excluded |
| 11 | Highest/lowest CV-ranked classifier confusion matrices | Pooled out-of-fold results, class-normalized percentages, TP/FN rates |
| 12 | Facies model comparison | Accuracy by well plus mean training-well CV accuracy |
| 13 | Actual facies and eight model tracks | Held-out Well_4; caption's model families preserved |
| 14 | Interaction/robust linear porosity comparison | Well_2 out-of-fold predictions, not fitted training predictions |
| 15 | Porosity model comparison | R² by well and mean training CV R² |
| 16 | Ten porosity model tracks | Held-out Well_4; caption's model families preserved |
| 17 | Squared-exponential GP/robust linear permeability | Well_2 out-of-fold predictions; no forced best/worst ordering |
| 18 | Permeability model comparison | R² by well and mean training CV R² |
| 19 | Nine permeability model tracks | Held-out Well_5; caption's model families preserved |

Each numbered figure is saved as PNG and vector PDF. All are labeled synthetic.
The previous 14 general-purpose plots remain available in `outputs/figures/`,
including all three target depth plots for **both** held-out wells.

## Model variants and scientific conventions

The manuscript's model counts, method names and text are not fully consistent.
This implementation retains the existing numbered registry (1–26) and adds the
variants explicitly named in figure captions:

| ID | Variant | Implementation |
|---:|---|---|
| 25 | Squared-exponential GP | Fixed RBF kernel |
| 27 | Robust linear | Huber regression, epsilon 1.35, up to 1000 iterations |
| 28 | Interaction linear | All pairwise predictor interactions plus main effects, ordinary least squares |
| 29 | Matérn GP | Fixed Matérn kernel, nu=1.5 |
| 30 | Rational-quadratic GP | Fixed rational-quadratic kernel, alpha=1 |
| 31 | Exponential GP | Fixed Matérn kernel, nu=0.5 |

All GP kernels use length scale sqrt(5), fixed signal variance 1, and regression
noise variance 0.05. They share the configurable training-row cap. The full registry
has 25 classifiers and 25 regressors for each of two targets: 75 task/model
combinations, each assessed across five wells. These are documented Python
implementations, not exact reproductions of MATLAB presets.

Porosity is a fraction; permeability is mD and is trained in log10 space. Errors
and R² are evaluated in original target units. Accuracy is used for nominal facies;
R² is not applied to arbitrary class codes. There is no invented regression
"accuracy". Negative R² values are preserved. Model families in Figures 13, 14,
16, 17 and 19 are fixed from manuscript captions, not chosen by test scores.

The synthetic generator uses layered facies, shared petrophysical latent variables,
well-to-well shifts and independent measurement noise. Laboratory measurements
are co-located with logs at every generated depth. These simplifying assumptions
make this a reproducible software demonstration, not evidence of field accuracy.
No scores, data counts, correlations or best-model claims from the manuscript are
hard-coded. Exact published images and numerical results require original source
data and geological graphics; Figures 1–2 are explicitly schematic substitutes.
