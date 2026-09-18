"""Generate data, validate by whole well, select models, then evaluate application wells."""
import argparse
import json
import platform
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from synthetic_data import generate, save_data, FEATURES, TARGETS, DEV_WELLS, APP_WELLS, FACIES
from models import make_model, NAMES, REG_IDS, GP_IDS
import plots


def metrics_for(y, pred, task):
    if task == 'Facies':
        return dict(Accuracy=accuracy_score(y,pred), BalancedAccuracy=balanced_accuracy_score(y,pred),
                    MacroF1=f1_score(y,pred, labels=range(4), average='macro', zero_division=0))
    return dict(R2=r2_score(y,pred), MSE=mean_squared_error(y,pred),
                RMSE=np.sqrt(mean_squared_error(y,pred)), MAE=mean_absolute_error(y,pred))


def fit_model(train, number, task, seed, gp_max, warning_log):
    fit = train
    if number in GP_IDS and len(train) > gp_max:
        # Only training rows are sampled; stratify by well AND facies for coverage.
        strata = train.Well + '_' + train.Facies.astype(str)
        take, _ = train_test_split(np.arange(len(train)), train_size=gp_max,
                                   stratify=strata, random_state=seed)
        fit = train.iloc[take]
    model = make_model(number, task, seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(fit[FEATURES], fit[task])
    for warning in caught:
        warning_log.append(dict(Model=number, Task=task, TrainingWells=sorted(train.Well.unique()),
                                Warning=str(warning.message)))
    return model, len(fit)


def record_predictions(frame, pred, number, task, split):
    result = frame[['Well','Depth_m']].copy()
    result['Task'], result['Model'], result['Split'] = task, number, split
    result['Truth'], result['Prediction'] = frame[task].to_numpy(), pred
    return result


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = generate(args.samples, args.seed)
    save_data(data, out/'data')
    dev = data[data.Well.isin(DEV_WELLS)].copy()
    application = data[data.Well.isin(APP_WELLS)].copy()
    registry = pd.DataFrame([dict(Model=n, Name=name, Facies=n <= 25,
        Regression=n in REG_IDS, Notes=('No direct regression equivalent' if n in [4,5,6,7,22,24]
        else 'Regression only; continuous targets, not facies codes' if n >= 26
        else 'Classifier and regression counterpart')) for n,name in NAMES.items()])
    registry.to_csv(out/'model_registry.csv', index=False)
    records, predictions, warning_log = [], [], []
    started = time.perf_counter()
    # Each development well is held out once. No adjacent-depth random split.
    for task in TARGETS:
        for number in (range(1,26) if task == 'Facies' else REG_IDS):
            print(f'CV {task:12s} {number:02d} {NAMES[number]}', flush=True)
            for well in DEV_WELLS:
                train, test = dev[dev.Well != well], dev[dev.Well == well]
                model, n_fit = fit_model(train, number, task, args.seed, args.gp_max, warning_log)
                pred = model.predict(test[FEATURES])
                records.append(dict(Task=task, Model=number, Name=NAMES[number], Well=well,
                    Split='development_cv', TrainingSamples=n_fit, **metrics_for(test[task],pred,task)))
                predictions.append(record_predictions(test,pred,number,task,'development_cv'))
    cv = pd.DataFrame(records)
    rankings = []
    selected = {}
    for task in TARGETS:
        metric = 'MacroF1' if task == 'Facies' else 'R2'
        rank = cv[cv.Task == task].groupby('Model')[metric].mean().reset_index(name='CVScore')
        rank = rank.sort_values(['CVScore','Model'], ascending=[False,True])
        rank['Task'], rank['SelectionMetric'] = task, metric
        selected[task] = int(rank.iloc[0].Model)
        rankings.append(rank)
    ranking = pd.concat(rankings, ignore_index=True)
    ranking.to_csv(out/'cv_ranking.csv', index=False)
    # Persist selection BEFORE any application labels are evaluated.
    (out/'selected_models.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    (out/'models').mkdir(exist_ok=True)
    for task in TARGETS:
        for number in (range(1,26) if task == 'Facies' else REG_IDS):
            print(f'Apply {task:12s} {number:02d} {NAMES[number]}', flush=True)
            model, n_fit = fit_model(dev,number,task,args.seed,args.gp_max,warning_log)
            if number == selected[task]:
                joblib.dump(dict(model=model, features=FEATURES, task=task,
                    number=number, name=NAMES[number], facies=FACIES), out/'models'/f'{task.lower()}.joblib')
            for well in APP_WELLS:
                test = application[application.Well == well]
                pred = model.predict(test[FEATURES])
                records.append(dict(Task=task, Model=number, Name=NAMES[number], Well=well,
                    Split='application', TrainingSamples=n_fit, **metrics_for(test[task],pred,task)))
                predictions.append(record_predictions(test,pred,number,task,'application'))
    metrics = pd.DataFrame(records)
    predictions = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(out/'metrics.csv', index=False)
    predictions.to_csv(out/'predictions_all_models.csv', index=False)
    chosen = application[['Well','Depth_m',*FEATURES]].copy().reset_index(drop=True)
    for task,number in selected.items():
        p = predictions[(predictions.Split == 'application') & (predictions.Task == task)
                        & (predictions.Model == number)]
        chosen = chosen.merge(p[['Well','Depth_m','Prediction']].rename(columns={'Prediction':task+'_predicted'}),
                              on=['Well','Depth_m'], validate='one_to_one')
    chosen['Facies_name'] = chosen.Facies_predicted.astype(int).map(dict(enumerate(FACIES)))
    chosen.to_csv(out/'application_predictions.csv', index=False)
    print('Rendering figures...', flush=True)
    figs = out/'figures'
    plots.logs(dev,figs)
    plots.correlation(dev,figs)
    plots.confusion_panels(predictions,selected,figs)
    plots.comparisons(metrics,figs)
    plots.regression_points(predictions,selected,figs)
    plots.tracks(predictions,ranking,figs)
    from manuscript_figures import render_all
    render_all(data, predictions, metrics, ranking, out/'manuscript_figures')
    manifest = dict(seed=args.seed, samples_per_well=args.samples, development_wells=DEV_WELLS,
        application_wells=APP_WELLS, gp_training_cap=args.gp_max, features=FEATURES,
        python=platform.python_version(), sklearn=sklearn.__version__,
        selected=selected, runtime_seconds=round(time.perf_counter()-started,2),
        warning_count=len(warning_log), synthetic=True)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'training_warnings.json').write_text(json.dumps(warning_log,indent=2),encoding='utf-8')
    summary = ['# Synthetic five-well experiment', '',
               'Models selected only by leave-one-well-out validation on Wells 1–3.',
               'Wells 4–5 are held-out application wells. All measurements are synthetic.', '']
    for task,number in selected.items():
        summary += [f'## {task}: {number} — {NAMES[number]}', '']
        for _,row in metrics[(metrics.Task == task)&(metrics.Model == number)&(metrics.Split == 'application')].iterrows():
            score = f'accuracy={row.Accuracy:.3f}, macro-F1={row.MacroF1:.3f}' if task == 'Facies' else f'R²={row.R2:.3f}, MAE={row.MAE:.4g}, MSE={row.MSE:.4g}'
            summary.append(f'- {row.Well}: {score}')
        summary.append('')
    (out/'RESULTS.md').write_text('\n'.join(summary),encoding='utf-8')
    print('\n'.join(summary),flush=True)
    print(f'Completed in {manifest["runtime_seconds"]} s. Output: {out.resolve()}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=360, help='Depth samples per well; minimum 120')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gp-max', type=int, default=240, help='Maximum training rows for exact Gaussian processes')
    parser.add_argument('--output', default='outputs')
    args = parser.parse_args()
    if args.samples < 120 or args.gp_max < 40:
        parser.error('--samples must be >=120 and --gp-max >=40')
    with threadpool_limits(limits=1):
        run(args)
