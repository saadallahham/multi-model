"""Rebuild the 19 manuscript figure types using computed synthetic results.

Run after run_experiment.py, or call render_all from that workflow.
Figures 1 and 2 are original conceptual schematics, not Enchova field data.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch
from plots import plt, save
from synthetic_data import FEATURES, FACIES, COLORS, DEV_WELLS, APP_WELLS
from models import NAMES

TITLES = [
    'Field location and depth map (synthetic schematic)',
    'Stratigraphic context (conceptual schematic)',
    'Well logs, facies and laboratory measurements',
    'Observations per well', 'Facies counts', 'Facies frequency by well',
    'Study workflow', 'Neutron-density crossplot', 'Predictor pairplot by facies',
    'Covariance and Pearson correlation', 'Development confusion matrices',
    'Facies model comparison', 'Facies depth tracks', 'Porosity model point comparisons',
    'Porosity model comparison', 'Porosity depth tracks',
    'Permeability model point comparisons', 'Permeability model comparison',
    'Permeability depth tracks']


def finish(fig, folder, number, suffix=''):
    fig.suptitle(f'Figure {number}. {TITLES[number-1]}{suffix}', fontsize=12)
    fig.text(.995, -.025, 'SYNTHETIC DEMONSTRATION', ha='right', fontsize=7, color='#555555')
    save(fig, folder, f'figure_{number:02d}')


def context_figures(folder):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    x, y = np.meshgrid(np.linspace(0, 12, 100), np.linspace(0, 10, 100))
    coast = 2 + .4*np.sin(np.linspace(0, 10, 100))
    axs[0].set_facecolor('#dcecf5')
    axs[0].fill_betweenx(np.linspace(0, 10, 100), 0, coast, color='#d9c298')
    axs[0].text(.6, 5, 'Land', rotation=90)
    axs[0].text(4, 8, 'Offshore synthetic field')
    axs[0].add_patch(Rectangle((6, 2), 4, 5, fill=False, linestyle='--', edgecolor='red'))
    positions = [(7, 3), (8, 4), (8.5, 6), (9, 3.5), (7, 6)]
    for ax in axs:
        for i, (wx, wy) in enumerate(positions):
            ax.scatter(wx, wy, c='#006c4c' if i < 3 else '#be4a2f', s=40, zorder=4, edgecolors='white')
            ax.annotate(f'W{i+1}', (wx, wy), xytext=(5, 5), textcoords='offset points')
        ax.set(xlabel='Local x (km)', ylabel='Local y (km)', xlim=(0, 12), ylim=(0, 10))
    depth = 2300 + 45*x + 15*y - 250*np.exp(-((x-8)**2/6+(y-5)**2/12))
    im = axs[1].contourf(x, y, depth, levels=14, cmap='viridis_r')
    axs[1].contour(x, y, depth, levels=10, colors='k', linewidths=.4)
    fig.colorbar(im, ax=axs[1], label='Synthetic structural depth (m)')
    axs[0].set_title('(A) Conceptual location; no real coordinates')
    axs[1].set_title('(B) Analytic surface; no measured structure')
    finish(fig, folder, 1)

    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    units = [('Younger cover', 'Fine marine sediments', '#d8dcde', '..'),
             ('Upper carbonate interval', 'Marl and lime mudstone', '#bdcbad', '--'),
             ('Reservoir interval', 'Grainstone and packstone', '#e8d790', 'oo'),
             ('Lower carbonate interval', 'Packstone and wackestone', '#ccd6a6', '//'),
             ('Underlying interval', 'Evaporitic substrate', '#d5bfdc', 'xx')]
    for i, (unit, lith, color, hatch) in enumerate(units):
        ax.add_patch(Rectangle((.25, i), 2, 1, facecolor=color, edgecolor='gray', hatch=hatch))
        ax.text(2.5, i+.5, unit, va='center', weight='bold')
        ax.text(6, i+.5, lith, va='center')
    ax.set(xlim=(0, 10.5), ylim=(5.2, -.6))
    ax.axis('off')
    ax.text(.25, -.25, 'Lithology', weight='bold')
    ax.text(2.5, -.25, 'Conceptual unit (youngest at top)', weight='bold')
    ax.text(6, -.25, 'Depositional description', weight='bold')
    finish(fig, folder, 2, '\nIllustrative carbonate succession; not a reconstruction of the published chart')


def input_figures(data, folder):
    well = data[data.Well == 'Well_2']
    fig, axs = plt.subplots(1, 8, figsize=(16, 7), sharey=True, layout='constrained')
    cols = ['GR', 'RHOB', 'NPHI', 'DT', 'RT_ohm_m', 'Facies', 'Porosity', 'Permeability']
    units = ['API', 'g/cm³', 'fraction', 'µs/ft', 'ohm m', 'class', 'fraction', 'mD']
    for ax, col, unit in zip(axs, cols, units):
        if col == 'Facies':
            ax.imshow(well.Facies.to_numpy()[:, None], aspect='auto', cmap=ListedColormap(COLORS),
                      vmin=-.5, vmax=3.5, extent=[0, 1, well.Depth_m.max(), well.Depth_m.min()])
            ax.set_xticks([])
        else:
            ax.plot(well[col], well.Depth_m, lw=.8, color='#176a52')
            if col == 'RT_ohm_m':
                ax.set_xscale('log')
        ax.set_title(f'{col}\n({unit})', fontsize=9)
        ax.set_ylim(well.Depth_m.max(), well.Depth_m.min())
        ax.grid(alpha=.2)
        ax.tick_params(axis='x', labelrotation=40, labelsize=8)
    axs[0].set_ylabel('Depth (m)')
    fig.legend(handles=[Patch(color=c, label=f) for c, f in zip(COLORS, FACIES)],
               loc='outside lower center', ncol=4)
    finish(fig, folder, 3, ' | Well_2')
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    counts = data.groupby('Well').size()
    bars = ax.bar(counts.index, counts, color=['#176a52']*3+['#bd623e']*2)
    ax.bar_label(bars)
    ax.set(ylabel='Observations', ylim=(0, counts.max()*1.15), xlabel='Wells 1–3: training; wells 4–5: test')
    finish(fig, folder, 4)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    counts = data.Facies.value_counts().reindex(range(4), fill_value=0)
    bars = ax.bar(FACIES, counts, color=COLORS)
    ax.bar_label(bars)
    ax.set(ylabel='Observations', ylim=(0, counts.max()*1.15))
    finish(fig, folder, 5)
    counts = pd.crosstab(data.Well, data.Facies).reindex(columns=range(4), fill_value=0)
    counts.columns = FACIES
    counts.to_csv(Path(folder)/'facies_counts.csv')
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    counts.plot.bar(ax=ax, color=COLORS, rot=0)
    ax.set(ylabel='Observations', xlabel='Well')
    ax.legend(ncol=2)
    finish(fig, folder, 6)


def workflow(folder):
    fig, ax = plt.subplots(figsize=(13, 4), layout='constrained')
    labels = ['Generate 5 synthetic wells\nLogs + laboratory targets',
              'Develop on wells 1–3\nLeave-one-well-out validation\nFit preprocessing within each fold',
              'Select by development scores\nMacro-F1 or R²\nRefit on all 3 training wells',
              'Test on wells 4–5\nCompute metrics and figures\nNo test-based model selection']
    for i, label in enumerate(labels):
        ax.text(i*.25+.12, .5, label, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=.7', fc='#e8f0ef', ec='#176a52'))
        if i < 3:
            ax.annotate('', xy=((i+1)*.25+.005, .5), xytext=(i*.25+.235, .5),
                        arrowprops=dict(arrowstyle='->', lw=2))
    ax.axis('off')
    finish(fig, folder, 7)


def crossplots(data, folder):
    dev = data[data.Well.isin(DEV_WELLS)]
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    for i, name in enumerate(FACIES):
        p = dev[(dev.Well == 'Well_2') & (dev.Facies == i)]
        ax.scatter(p.NPHI, p.RHOB, s=10, c=COLORS[i], label=name, alpha=.7)
    ax.set(xlabel='NPHI (fraction)', ylabel='RHOB (g/cm³)')
    ax.invert_yaxis()
    ax.grid(alpha=.25)
    ax.legend()
    finish(fig, folder, 8, ' | Well_2; colored by synthetic facies')
    fig, axs = plt.subplots(5, 5, figsize=(13, 12), layout='constrained')
    for i, y in enumerate(FEATURES):
        for j, x in enumerate(FEATURES):
            ax = axs[i, j]
            for k in range(4):
                p = dev[dev.Facies == k]
                if i == j:
                    grid = np.linspace(dev[x].min(), dev[x].max(), 160)
                    ax.plot(grid, gaussian_kde(p[x])(grid), color=COLORS[k], lw=1)
                else:
                    ax.scatter(p[x], p[y], s=2, color=COLORS[k], alpha=.3, rasterized=True)
            if i == 4: ax.set_xlabel(x)
            if j == 0: ax.set_ylabel(y if i != j else 'Density')
            ax.tick_params(labelsize=6)
    fig.legend(handles=[Patch(color=c, label=f) for c, f in zip(COLORS, FACIES)],
               loc='outside lower center', ncol=4)
    finish(fig, folder, 9, ' | training wells only')
    columns = FEATURES + ['Porosity', 'Permeability']
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    for ax, matrix, title in [(axs[0], dev[columns].cov(), 'Covariance (mixed units)'),
                               (axs[1], dev[columns].corr(), 'Pearson correlation')]:
        vmax = abs(matrix.to_numpy()).max() if title.startswith('Covariance') else 1
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        for i in range(len(columns)):
            for j in range(len(columns)):
                v = matrix.iloc[i, j]
                ax.text(j, i, f'{v:.2g}', ha='center', va='center', fontsize=8,
                        color='white' if abs(v) > .6*vmax else 'black')
        ax.set_xticks(range(len(columns)), columns, rotation=60, ha='right')
        ax.set_yticks(range(len(columns)), columns)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.8)
    finish(fig, folder, 10, '\nTraining wells only; nominal facies codes excluded')


def confusion(predictions, ranking, folder):
    ids = ranking[ranking.Task == 'Facies'].Model.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, number, label in zip(axs, [ids[0], ids[-1]], ['Highest', 'Lowest']):
        p = predictions[(predictions.Split == 'development_cv') &
                        (predictions.Task == 'Facies') & (predictions.Model == number)]
        cm = confusion_matrix(p.Truth, p.Prediction, labels=range(4), normalize='true')
        values = np.column_stack([cm, np.diag(cm), 1-np.diag(cm)])
        ax.imshow(values, cmap='Greens', vmin=0, vmax=1, aspect='auto')
        for i in range(4):
            for j in range(6):
                ax.text(j, i, f'{values[i,j]:.1%}', ha='center', va='center', fontsize=9,
                        color='white' if values[i,j] > .65 else 'black')
        ax.set_xticks(range(6), ['M', 'PW', 'GP', 'G', 'TP', 'FN'])
        ax.set_yticks(range(4), ['M', 'PW', 'GP', 'G'])
        ax.set(xlabel='Predicted class / per-class rate', ylabel='True class',
               title=f'{label} mean CV macro-F1: {NAMES[number]}\nPooled CV accuracy = {accuracy_score(p.Truth,p.Prediction):.3f}')
        ax.axvline(3.5, color='black')
    finish(fig, folder, 11, '\nOut-of-fold predictions from training wells; M / PW / GP / G follow the facies legend')


def comparison(metrics, task, folder, number):
    metric = 'Accuracy' if task == 'Facies' else 'R2'
    table = metrics[metrics.Task == task].pivot(index='Model', columns='Well', values=metric).sort_index()
    fig, ax = plt.subplots(figsize=(15, 6), layout='constrained')
    x = np.arange(len(table))
    for j, well in enumerate(table.columns):
        ax.bar(x+(j-2)*.16, table[well], .16, label=well+(' CV' if well in DEV_WELLS else ' test'))
    ax.plot(x, table[DEV_WELLS].mean(axis=1), 'k.-', label='Mean development CV')
    ax.set_xticks(x, table.index)
    ax.set(xlabel='Model ID (model_registry.csv)', ylabel='Accuracy' if task == 'Facies' else 'R²')
    ax.axhline(0, color='gray', lw=.6)
    fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center', ncol=3)
    finish(fig, folder, number)


def point_comparison(predictions, task, ids, folder, number):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, model_id in zip(axs, ids):
        p = predictions[(predictions.Well == 'Well_2') & (predictions.Task == task) &
                        (predictions.Model == model_id)].sort_values('Depth_m')
        ax.scatter(np.arange(len(p)), p.Truth, s=8, c='#1769bc', label='Synthetic laboratory')
        ax.scatter(np.arange(len(p)), p.Prediction, s=8, c='#eba600', label='Out-of-fold estimate')
        stats = f'R²={r2_score(p.Truth,p.Prediction):.3f}\nMSE={mean_squared_error(p.Truth,p.Prediction):.4g}\nMAE={mean_absolute_error(p.Truth,p.Prediction):.4g}'
        ax.text(.98, .98, stats, transform=ax.transAxes, ha='right', va='top',
                bbox=dict(fc='white', ec='none', alpha=.8))
        ax.set(title=NAMES[model_id], xlabel='Depth sample index',
               ylabel=task+(' (fraction)' if task == 'Porosity' else ' (mD)'))
        ax.grid(alpha=.2)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=2)
    finish(fig, folder, number, ' | Well_2 development validation')


def depth_tracks(predictions, task, ids, folder, number, well):
    p = predictions[(predictions.Well == well) & (predictions.Task == task)]
    facies = task == 'Facies'
    fig, axs = plt.subplots(1, len(ids)+int(facies), figsize=(17, 8), sharey=True, layout='constrained')
    truth = p[p.Model == ids[0]].sort_values('Depth_m')
    extent = [0, 1, truth.Depth_m.max(), truth.Depth_m.min()]
    if facies:
        axs[0].imshow(truth.Truth.to_numpy()[:, None], extent=extent, aspect='auto',
                      cmap=ListedColormap(COLORS), vmin=-.5, vmax=3.5, interpolation='nearest')
        axs[0].set_title('Laboratory', fontsize=9)
        axs[0].set_xticks([])
    low, high = p[p.Model.isin(ids)][['Truth', 'Prediction']].min().min(), p[p.Model.isin(ids)][['Truth', 'Prediction']].max().max()
    for ax, model_id in zip(axs[int(facies):], ids):
        q = p[p.Model == model_id].sort_values('Depth_m')
        if facies:
            ax.imshow(q.Prediction.to_numpy()[:, None], extent=extent, aspect='auto',
                      cmap=ListedColormap(COLORS), vmin=-.5, vmax=3.5, interpolation='nearest')
            ax.set_xticks([])
            label = f'Acc={accuracy_score(q.Truth,q.Prediction):.3f}'
        else:
            ax.plot(q.Truth, q.Depth_m, color='#a83f32', lw=.8, label='Laboratory')
            ax.plot(q.Prediction, q.Depth_m, color='#086f55', lw=.8, label='Estimate')
            ax.set_xlim(low-.03*(high-low), high+.03*(high-low))
            ax.tick_params(axis='x', rotation=60, labelsize=7)
            label = f'R²={r2_score(q.Truth,q.Prediction):.3f}'
        ax.set_title(NAMES[model_id].replace(' ', '\n'), fontsize=8)
        ax.set_xlabel(label, fontsize=8)
        ax.grid(alpha=.2)
    axs[0].set_ylim(extent[2], extent[3])
    axs[0].set_ylabel('Depth (m)')
    if facies:
        fig.legend(handles=[Patch(color=c, label=f) for c, f in zip(COLORS,FACIES)], loc='outside lower center', ncol=4)
    else:
        fig.legend(*axs[-1].get_legend_handles_labels(), loc='outside lower center', ncol=2)
    finish(fig, folder, number, f' | {well} held-out test'+('' if facies else ' | '+('fraction' if task == 'Porosity' else 'mD')))


def render_all(data, predictions, metrics, ranking, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    context_figures(folder)
    input_figures(data, folder)
    workflow(folder)
    crossplots(data, folder)
    confusion(predictions, ranking, folder)
    comparison(metrics, 'Facies', folder, 12)
    depth_tracks(predictions, 'Facies', [20, 1, 24, 8, 3, 2, 21, 7], folder, 13, 'Well_4')
    point_comparison(predictions, 'Porosity', [28, 27], folder, 14)
    comparison(metrics, 'Porosity', folder, 15)
    depth_tracks(predictions, 'Porosity', [27, 8, 21, 13, 29, 25, 30, 20, 3, 31], folder, 16, 'Well_4')
    point_comparison(predictions, 'Permeability', [25, 27], folder, 17)
    comparison(metrics, 'Permeability', folder, 18)
    depth_tracks(predictions, 'Permeability', [20, 2, 1, 21, 3, 31, 30, 29, 13], folder, 19, 'Well_5')
    entries = [dict(number=i, title=t, file=f'figure_{i:02d}.png',
                    kind='conceptual schematic' if i <= 2 else 'computed synthetic data') for i,t in enumerate(TITLES,1)]
    (folder/'figure_manifest.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
    html = '<!doctype html><meta charset="utf-8"><title>Synthetic manuscript figures</title><style>body{font-family:Arial;max-width:1200px;margin:40px auto}img{width:100%;border:1px solid #ddd}section{margin-bottom:48px}</style><h1>Synthetic manuscript figures</h1><p>Three training wells and two held-out test wells. Figures 1–2 are conceptual schematics; all other figures use generated data and computed predictions.</p>'
    html += ''.join(f'<section><h2>Figure {e["number"]}: {e["title"]}</h2><a href="figure_{e["number"]:02d}.pdf">Vector PDF</a><img src="{e["file"]}" alt="{e["title"]}"></section>' for e in entries)
    (folder/'index.html').write_text(html, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs', help='Existing experiment output directory')
    args = parser.parse_args()
    out = Path(args.output)
    logs = pd.read_csv(out/'data/application_logs.csv')
    truth = pd.read_csv(out/'data/application_truth_for_evaluation_only.csv')
    data = pd.concat([pd.read_csv(out/'data/development.csv'), logs.merge(truth, on=['Well','Depth_m'], validate='one_to_one')])
    render_all(data, pd.read_csv(out/'predictions_all_models.csv'), pd.read_csv(out/'metrics.csv'),
               pd.read_csv(out/'cv_ranking.csv'), out/'manuscript_figures')
