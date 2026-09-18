"""Apply saved selected models to logs only; no laboratory targets are needed."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from synthetic_data import FEATURES, TARGETS


def apply(input_csv, model_dir, output_csv):
    data = pd.read_csv(input_csv)
    required = ['Well', 'Depth_m', *FEATURES]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if data.empty or data.duplicated(['Well','Depth_m']).any():
        raise ValueError('Input must have rows and unique (Well, Depth_m) pairs')
    if np.isinf(data[FEATURES].to_numpy(dtype=float)).any():
        raise ValueError('Predictors cannot contain infinity; use NaN for missing measurements')
    for task in TARGETS:
        bundle = joblib.load(Path(model_dir)/f'{task.lower()}.joblib')
        data[task+'_predicted'] = bundle['model'].predict(data[bundle['features']])
        if task == 'Facies':
            data['Facies_name'] = data.Facies_predicted.astype(int).map(dict(enumerate(bundle['facies'])))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_csv,index=False)
    print(f'Saved {len(data)} predictions to {output_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='outputs/data/application_logs.csv')
    parser.add_argument('--models', default='outputs/models')
    parser.add_argument('--output', default='outputs/reapplied_predictions.csv')
    args = parser.parse_args()
    apply(args.input,args.models,args.output)
