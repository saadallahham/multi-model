"""Illustrative synthetic carbonate logs and co-located core measurements."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ['DT', 'GR', 'NPHI', 'RHOB', 'LogRT']
TARGETS = ['Facies', 'Porosity', 'Permeability']
DEV_WELLS = ['Well_1', 'Well_2', 'Well_3']
APP_WELLS = ['Well_4', 'Well_5']
FACIES = ['Mudstone', 'Packstone/wackestone', 'Grainstone/packstone', 'Grainstone']
COLORS = ['#00d855', '#eadb00', '#1769ff', '#7800c8']


def generate(n=360, seed=42):
    if n < 120:
        raise ValueError('Use at least 120 depth samples per well.')
    frames = []
    for w in range(5):
        rng = np.random.default_rng(seed + w * 1009)
        depth = np.linspace(120, 360, n)
        t = np.linspace(0, 1, n)
        strat = t + 0.025 * np.sin(4 * np.pi * t + w) + rng.normal(0, .002, n)
        bounds = np.array([.14, .30, .35, .49, .63, .68, .82, .92])
        sequence = np.array([0, 1, 2, 3, 1, 0, 2, 3, 1])
        facies = sequence[np.searchsorted(bounds, strat)]
        # Thin beds and well-to-well variation prevent an identical stratigraphy.
        thin = rng.random(n) < .045
        facies[thin] = rng.integers(0, 4, thin.sum())
        base_phi = np.array([.09, .17, .23, .28])[facies]
        phi = np.clip(base_phi + .022*np.sin(18*np.pi*t + w*.3)
                      + rng.normal(0, .009, n) + rng.normal(0, .008), .025, .37)
        clay = np.clip(np.array([.65, .30, .12, .04])[facies]
                       + rng.normal(0, .045, n), .005, .9)
        logk = -2.0 + 11.5*phi + np.array([-.35, -.1, .15, .35])[facies]
        logk += rng.normal(0, .11, n)
        # Measurement noise is separate for logs and simulated laboratory assays.
        lab_phi = np.clip(phi + rng.normal(0, .004, n), .005, .45)
        lab_k = 10 ** (logk + rng.normal(0, .05, n))
        dt = 48 + 155*phi + 12*clay + rng.normal(0, 2.2, n)
        gr = 18 + 115*clay + rng.normal(0, 5, n) + rng.normal(0, 2)
        nphi = phi + .07*clay + rng.normal(0, .009, n)
        rhob = 2.71*(1-phi) + 1.05*phi - .10*clay + rng.normal(0, .013, n)
        logrt = .55 - 1.65*np.log10(phi) - 1.0*clay + rng.normal(0, .10, n)
        frames.append(pd.DataFrame(dict(Well=f'Well_{w+1}', Depth_m=depth,
            DT=dt, GR=gr, NPHI=nphi, RHOB=rhob, LogRT=logrt,
            RT_ohm_m=10**logrt, Facies=facies, Porosity=lab_phi, Permeability=lab_k)))
    return pd.concat(frames, ignore_index=True)


def save_data(frame, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for well, data in frame.groupby('Well'):
        data[['Well', 'Depth_m', *FEATURES, 'RT_ohm_m']].to_csv(folder/f'{well}_logs.csv', index=False)
        data[['Well', 'Depth_m', *TARGETS]].to_csv(folder/f'{well}_laboratory.csv', index=False)
    frame[frame.Well.isin(DEV_WELLS)].to_csv(folder/'development.csv', index=False)
    frame[frame.Well.isin(APP_WELLS)][['Well', 'Depth_m', *FEATURES, 'RT_ohm_m']].to_csv(
        folder/'application_logs.csv', index=False)
    frame[frame.Well.isin(APP_WELLS)][['Well', 'Depth_m', *TARGETS]].to_csv(
        folder/'application_truth_for_evaluation_only.csv', index=False)
