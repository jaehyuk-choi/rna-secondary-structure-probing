#!/usr/bin/env python3
"""VL0 F1 vs α from vl0_alpha_sweep_both.csv; Vienna and Contrafold subplots."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / 'figures' / 'main'
DATA_PATH = REPO_ROOT / 'results' / 'sweeps' / 'vl0_alpha_sweep_both.csv'

MODEL_ORDER = ['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert']
MODEL_LABELS = {'ernie': 'ERNIE', 'roberta': 'RoBERTa', 'rnafm': 'RNAFM',
                'rinalmo': 'RiNALMo', 'onehot': 'One-hot', 'rnabert': 'RNABERT'}
LINE_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

FONT_TITLE = 14
FONT_AXIS = 12
FONT_TICK = 11
FONT_LEGEND = 11

# RGB(255, 245, 242)
LIGHT_ORANGE = (255/255, 245/255, 242/255)


def load_sweep_data():
    """Return {backend: {model: {alpha: f1}}}."""
    data = {'Vienna': {m: {} for m in MODEL_ORDER},
            'Contrafold': {m: {} for m in MODEL_ORDER}}
    with open(DATA_PATH) as f:
        for r in csv.DictReader(f):
            alpha = float(r['alpha'])
            for model in MODEL_ORDER:
                for backend in ['Vienna', 'Contrafold']:
                    col = f'{model}_{backend}'
                    val = r.get(col, '').strip()
                    if val:
                        data[backend][model][alpha] = float(val)
    return data


def get_best_alpha(alpha_f1):
    """Return (alpha*, f1*) for max F1."""
    if not alpha_f1:
        return 0, 0
    best_a, best_f1 = max(alpha_f1.items(), key=lambda x: x[1])
    return best_a, best_f1


def plot_single_backend(ax, sweep, backend_name):
    """Plot F1 vs α for one backend on the given axes."""
    for i, model in enumerate(MODEL_ORDER):
        alpha_f1 = sweep[model]
        if not alpha_f1:
            continue
        alphas = sorted(alpha_f1.keys())
        f1s = [alpha_f1[a] for a in alphas]
        best_a, best_f1 = get_best_alpha(alpha_f1)
        ax.plot(alphas, f1s, '-',
                label=f"{MODEL_LABELS.get(model, model)} ({best_a:.2f})",
                color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=2)
        idx = min(range(len(alphas)), key=lambda j: abs(alphas[j] - best_a))
        ax.scatter([alphas[idx]], [f1s[idx]], marker='*', s=300,
                   color=LINE_COLORS[i % len(LINE_COLORS)],
                   edgecolors='black', linewidths=0.5, zorder=5)

    ax.set_xlabel('α (CPLfold bonus weight)', fontsize=FONT_AXIS, fontweight='bold')
    ax.set_ylabel('Mean F1', fontsize=FONT_AXIS, fontweight='bold')
    ax.set_title(f'Validation (VL0): F1 vs α — {backend_name}',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.legend(loc='upper right', ncol=2, fontsize=FONT_LEGEND, prop=dict(weight='bold'))
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0.5, 0.75)
    ax.grid(alpha=0.3)


def fig_single(data, backend):
    """Standalone single-backend figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_single_backend(ax, data[backend], backend)
    plt.tight_layout()
    out = FIG_DIR / f'vl0_alpha_sweep_{backend.lower()}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def fig_both(data, facecolor='white', suffix=''):
    """Side-by-side Vienna + Contrafold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor(facecolor)
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    for ax, backend in [(ax1, 'Vienna'), (ax2, 'Contrafold')]:
        for i, model in enumerate(MODEL_ORDER):
            alpha_f1 = data[backend][model]
            if not alpha_f1:
                continue
            alphas = sorted(alpha_f1.keys())
            f1s = [alpha_f1[a] for a in alphas]
            ax.plot(alphas, f1s, 'o-', label=MODEL_LABELS.get(model, model),
                    color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=2,
                    markersize=3, markevery=5)
        ax.set_xlabel('α')
        ax.set_ylabel('Mean F1')
        ax.set_title(f'VL0 — {backend}')
        ax.legend(loc='upper right', ncol=2)
        ax.set_xlim(0, 2.05)
        ax.set_ylim(0.5, 0.75)
        ax.grid(alpha=0.3)

    plt.suptitle('Validation (VL0): F1 vs α (0→2, step 0.02)', fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / f'vl0_alpha_sweep_both{suffix}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=facecolor)
    plt.close()
    print(f"Saved {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_sweep_data()
    fig_single(data, 'Vienna')
    fig_single(data, 'Contrafold')
    fig_both(data, facecolor='white')
    fig_both(data, facecolor=LIGHT_ORANGE, suffix='_lightorange')
    print("figures ->", FIG_DIR)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
