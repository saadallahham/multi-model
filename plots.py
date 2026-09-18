"""Reference-inspired figures; every annotation is calculated from predictions."""
from pathlib import Path
import os
import textwrap
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
from synthetic_data import FEATURES, FACIES, COLORS, APP_WELLS
from models import NAMES

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 180, 'font.family': 'DejaVu Sans'})


def save(fig, folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder/f'{name}.png', bbox_inches='tight')
    fig.savefig(folder/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def correlation(data, folder):
    # Facies codes are nominal: intentionally not assigned a Pearson coefficient.
    columns = [*FEATURES, 'Porosity', 'Permeability']
    corr = data[columns].corr()
    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=55, ha='right')
    ax.set_yticks(range(len(columns)), columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            val = corr.iloc[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color='white' if abs(val) > .65 else 'black')
    fig.colorbar(im, ax=ax, label='Pearson correlation')
    ax.set_title('Synthetic development wells | continuous variables')
    save(fig, folder, '01_correlation')


def confusion_panels(predictions, selected, folder):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    cmap = plt.get_cmap('Greens')
    for ax, well in zip(axes, APP_WELLS):
        p = predictions[(predictions.Well == well) & (predictions.Task == 'Facies')
                        & (predictions.Model == selected['Facies'])]
        cm = confusion_matrix(p.Truth, p.Prediction, labels=range(4), normalize='true')
        values = np.column_stack([cm, np.diag(cm), 1-np.diag(cm)])
        rgba = np.ones((4, 6, 4))
        for i in range(4):
            for j in range(6):
                good = j == i or j == 4
                rgba[i, j] = (cmap if good else plt.get_cmap('Reds'))(.08+.85*values[i, j])
        ax.imshow(rgba, aspect='auto')
        for i in range(4):
            for j in range(6):
                ax.text(j, i, f'{100*values[i,j]:.1f}%', ha='center', va='center', fontsize=9,
                        color='white' if values[i,j] > .7 else 'black')
        ax.set_xticks(range(6), ['M', 'PW', 'GP', 'G', 'TP', 'FN'])
        ax.set_yticks(range(4), ['M', 'PW', 'GP', 'G'])
        ax.axvline(3.5, color='black', lw=2)
        ax.set(xlabel='Predicted facies | per-class rates', ylabel='True facies',
               title=f'{well} | {NAMES[selected["Facies"]]}')
    fig.suptitle('Held-out synthetic wells | confusion matrices (%)\n'
                 'M: mudstone; PW: packstone/wackestone; GP: grainstone/packstone; G: grainstone', fontsize=11)
    save(fig, folder, '02_confusion_matrices')


def comparisons(metrics, folder):
    for task, name in [('Facies', '03_facies_model_comparison'),
                       ('Porosity', '04_porosity_model_comparison'),
                       ('Permeability', '05_permeability_model_comparison')]:
        data = metrics[metrics.Task == task]
        value = 'Accuracy' if task == 'Facies' else 'R2'
        table = data.pivot(index='Model', columns='Well', values=value).sort_index()
        fig, ax = plt.subplots(figsize=(15, 5), layout='constrained')
        x = np.arange(len(table))
        for j, well in enumerate(table.columns):
            vals = table[well].to_numpy() * (100 if task == 'Facies' else 1)
            ax.bar(x + (j-2)*.16, vals, .16, label=well + (' (CV)' if well in ['Well_1','Well_2','Well_3'] else ' (held out)'))
        dev = table[['Well_1','Well_2','Well_3']].mean(axis=1) * (100 if task == 'Facies' else 1)
        ax.plot(x, dev, 'k.-', label='Mean development CV', lw=1.8)
        ax.set_xticks(x, table.index)
        ax.set(xlabel='Model number (see model_registry.csv)',
               ylabel='Accuracy (%)' if task == 'Facies' else 'R²',
               title=f'{task}: cross-well validation and held-out application')
        if task == 'Facies':
            ax.set_ylim(0, 105)
        else:
            ax.axhline(0, color='gray', lw=.6)
        ax.grid(axis='y', alpha=.2)
        ax.legend(ncol=3, fontsize=9, loc='upper center', bbox_to_anchor=(.5, -.15))
        save(fig, folder, name)


def regression_points(predictions, selected, folder):
    for task, name, unit in [('Porosity','06_porosity_points','fraction'),
                              ('Permeability','07_permeability_points','mD')]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
        for ax, well in zip(axes, APP_WELLS):
            p = predictions[(predictions.Well == well) & (predictions.Task == task)
                            & (predictions.Model == selected[task])].sort_values('Depth_m')
            ax.scatter(np.arange(len(p)), p.Truth, s=9, color='#006bbb', label='Synthetic laboratory')
            ax.scatter(np.arange(len(p)), p.Prediction, s=8, color='#eca900', label='Estimated')
            annotation = (f'R² = {r2_score(p.Truth, p.Prediction):.3f}\n'
                          f'MSE = {mean_squared_error(p.Truth, p.Prediction):.4g}\n'
                          f'MAE = {mean_absolute_error(p.Truth, p.Prediction):.4g}')
            ax.text(.97, .97, annotation, transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
            ax.set(xlabel='Depth sample index', ylabel=f'{task} ({unit})', title=well)
            ax.grid(alpha=.25)
            ax.legend(loc='lower center', bbox_to_anchor=(.5,-.26), ncol=2)
        fig.suptitle(f'Selected {task.lower()} model: {selected[task]} — {NAMES[selected[task]]}')
        save(fig, folder, name)


def tracks(predictions, ranking, folder, top=8):
    for well in APP_WELLS:
        for task in ['Facies', 'Porosity', 'Permeability']:
            ids = ranking[ranking.Task == task].Model.head(top).tolist()
            data = predictions[(predictions.Well == well) & (predictions.Task == task)]
            truth = data[data.Model == ids[0]].sort_values('Depth_m')
            is_class = task == 'Facies'
            fig, axes = plt.subplots(1, len(ids)+int(is_class),
                figsize=(max(12, 1.75*(len(ids)+int(is_class))), 8), sharey=True, layout='constrained')
            extent = [0, 1, truth.Depth_m.max(), truth.Depth_m.min()]
            if is_class:
                axes[0].imshow(truth.Truth.to_numpy()[:,None], cmap=ListedColormap(COLORS),
                               vmin=-.5, vmax=3.5, aspect='auto', extent=extent, interpolation='nearest')
                axes[0].set_title('Laboratory', fontsize=10)
                axes[0].set_xticks([])
                axes[0].set_ylim(extent[2], extent[3])
            for ax, number in zip(axes[int(is_class):], ids):
                p = data[data.Model == number].sort_values('Depth_m')
                if is_class:
                    ax.imshow(p.Prediction.to_numpy()[:,None], cmap=ListedColormap(COLORS),
                              vmin=-.5, vmax=3.5, aspect='auto', extent=extent, interpolation='nearest')
                    ax.set_xticks([])
                    score = f'Acc={100*np.mean(p.Truth == p.Prediction):.1f}%'
                else:
                    ax.plot(p.Truth, p.Depth_m, color='#bd2929', lw=1, label='Synthetic laboratory')
                    ax.plot(p.Prediction, p.Depth_m, color='#006c4c', lw=.9, label='Estimate')
                    ax.set_xlim(data[['Truth','Prediction']].min().min()*.95,
                                data[['Truth','Prediction']].max().max()*1.05)
                    ax.tick_params(axis='x', labelsize=8, rotation=45)
                    score = f'R²={r2_score(p.Truth, p.Prediction):.3f}'
                ax.set_ylim(extent[2], extent[3])
                ax.set_title(f'{number}\n' + textwrap.fill(NAMES[number], 18), fontsize=8)
                ax.set_xlabel(score, fontsize=9)
                ax.grid(alpha=.25)
            axes[0].set_ylabel('Depth (m)')
            handles = [Patch(color=c, label=n) for c,n in zip(COLORS,FACIES)] if is_class else axes[-1].get_legend_handles_labels()[0]
            labels = FACIES if is_class else ['Synthetic laboratory','Estimate']
            fig.legend(handles, labels, loc='outside lower center', ncol=4 if is_class else 2, fontsize=9)
            fig.suptitle(f'{well} | {task}' + (' (fraction)' if task == 'Porosity' else ' (mD)' if task == 'Permeability' else '')
                         + ' | models ranked using development wells only')
            save(fig, folder, f'08_tracks_{well}_{task.lower()}')


def logs(data, folder):
    well = data[data.Well == 'Well_1']
    fig, axes = plt.subplots(1, 5, figsize=(12, 7), sharey=True, layout='constrained')
    labels = ['DT (µs/ft)', 'GR (API)', 'NPHI (fraction)', 'RHOB (g/cm³)', 'log10 RT (Ω m)']
    for ax, col, label in zip(axes, FEATURES, labels):
        ax.plot(well[col], well.Depth_m, lw=.9)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=.25)
    axes[0].invert_yaxis()
    axes[0].set_ylabel('Depth (m)')
    fig.suptitle('Well_1 | synthetic input logs')
    save(fig, folder, '00_input_logs')
