#!/usr/bin/env python3
"""
Validation (VL0) set: F1 vs α sweep (0 to 2.0, step 0.02).

Reads feb23 results_vl0_feb8 (Vienna) and results_vl0_contrafold_feb8 (Contrafold).
Plots mean F1 per alpha for each model.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MARCH1 = Path(__file__).resolve().parents[1]
FIG_DIR = MARCH1 / 'figures'

# feb23 VL0 results (alpha sweep 0~2, step 0.02)
VL0_VIENNA = Path('/projects/u6cg/jay/dissertations/feb23/results_vl0_feb8')
VL0_CONTRA = Path('/projects/u6cg/jay/dissertations/feb23/results_vl0_contrafold_feb8')

MODEL_ORDER = ['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert']
MODEL_LABELS = {'ernie': 'ERNIE', 'roberta': 'RoBERTa', 'rnafm': 'RNAFM',
                'rinalmo': 'RiNALMo', 'onehot': 'One-hot', 'rnabert': 'RNABERT'}
LINE_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

# Font sizes: title/axis labels bold, all text slightly larger
FONT_TITLE = 14
FONT_AXIS = 12
FONT_TICK = 11
FONT_LEGEND = 11


def load_mean_f1_per_alpha(csv_path):
    """Return {alpha: mean_f1}."""
    if not csv_path.exists():
        return {}
    by_alpha = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            by_alpha[float(r['alpha'])].append(float(r['f1']))
    return {a: np.mean(vals) for a, vals in sorted(by_alpha.items())}


def get_best_alpha(data):
    """Return (alpha*, f1*) for max F1."""
    if not data:
        return 0, 0
    best_a, best_f1 = max(data.items(), key=lambda x: x[1])
    return best_a, best_f1


def fig_vl0_vienna():
    """VL0 Vienna: F1 vs α (0~2, step 0.02) per model. Star at best α, legend shows α*."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(MODEL_ORDER):
        path = VL0_VIENNA / f'detailed_results_{model}.csv'
        data = load_mean_f1_per_alpha(path)
        if not data:
            continue
        alphas = sorted(data.keys())
        f1s = [data[a] for a in alphas]
        best_a, best_f1 = get_best_alpha(data)
        ax.plot(alphas, f1s, '-', label=f"{MODEL_LABELS.get(model, model)} ({best_a:.2f})",
                color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=2)
        idx = min(range(len(alphas)), key=lambda j: abs(alphas[j] - best_a))
        ax.scatter([alphas[idx]], [f1s[idx]], marker='*', s=300,
                   color=LINE_COLORS[i % len(LINE_COLORS)], edgecolors='black',
                   linewidths=0.5, zorder=5)

    ax.set_xlabel('α (CPLfold bonus weight)', fontsize=FONT_AXIS, fontweight='bold')
    ax.set_ylabel('Mean F1', fontsize=FONT_AXIS, fontweight='bold')
    ax.set_title('Validation (VL0): F1 vs α — Vienna', fontsize=FONT_TITLE, fontweight='bold')
    ax.legend(loc='upper right', ncol=2, fontsize=FONT_LEGEND, prop=dict(weight='bold'))
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0.5, 0.75)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'vl0_alpha_sweep_vienna.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {FIG_DIR / 'vl0_alpha_sweep_vienna.png'}")


def fig_vl0_contrafold():
    """VL0 Contrafold: F1 vs α per model. Star at best α, legend shows α*."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(MODEL_ORDER):
        path = VL0_CONTRA / f'detailed_results_{model}.csv'
        data = load_mean_f1_per_alpha(path)
        if not data:
            continue
        alphas = sorted(data.keys())
        f1s = [data[a] for a in alphas]
        best_a, best_f1 = get_best_alpha(data)
        ax.plot(alphas, f1s, '-', label=f"{MODEL_LABELS.get(model, model)} ({best_a:.2f})",
                color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=2)
        idx = min(range(len(alphas)), key=lambda j: abs(alphas[j] - best_a))
        ax.scatter([alphas[idx]], [f1s[idx]], marker='*', s=300,
                   color=LINE_COLORS[i % len(LINE_COLORS)], edgecolors='black',
                   linewidths=0.5, zorder=5)

    ax.set_xlabel('α (CPLfold bonus weight)', fontsize=FONT_AXIS, fontweight='bold')
    ax.set_ylabel('Mean F1', fontsize=FONT_AXIS, fontweight='bold')
    ax.set_title('Validation (VL0): F1 vs α — Contrafold', fontsize=FONT_TITLE, fontweight='bold')
    ax.legend(loc='upper right', ncol=2, fontsize=FONT_LEGEND, prop=dict(weight='bold'))
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0.5, 0.75)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'vl0_alpha_sweep_contrafold.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {FIG_DIR / 'vl0_alpha_sweep_contrafold.png'}")


def fig_vl0_both():
    """VL0 Vienna + Contrafold in 2 subplots. White background, legend upper right (same layout as lightorange)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, base_dir, title in [
        (ax1, VL0_VIENNA, 'Vienna'),
        (ax2, VL0_CONTRA, 'Contrafold'),
    ]:
        for i, model in enumerate(MODEL_ORDER):
            path = base_dir / f'detailed_results_{model}.csv'
            data = load_mean_f1_per_alpha(path)
            if not data:
                continue
            alphas = sorted(data.keys())
            f1s = [data[a] for a in alphas]
            ax.plot(alphas, f1s, 'o-', label=MODEL_LABELS.get(model, model),
                    color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=2, markersize=3, markevery=5)
        ax.set_xlabel('α')
        ax.set_ylabel('Mean F1')
        ax.set_title(f'VL0 — {title}')
        ax.legend(loc='upper right', ncol=2)
        ax.set_xlim(0, 2.05)
        ax.set_ylim(0.5, 0.75)
        ax.grid(alpha=0.3)

    plt.suptitle('Validation (VL0): F1 vs α (0→2, step 0.02)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'vl0_alpha_sweep_both.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[INFO] Saved {FIG_DIR / 'vl0_alpha_sweep_both.png'}")


# RGB(255, 245, 242) = light orange background
LIGHT_ORANGE = (255/255, 245/255, 242/255)


def fig_vl0_both_lightorange():
    """VL0 Vienna + Contrafold, light orange figure bg, legend upper right, grid area white."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor(LIGHT_ORANGE)

    for ax, base_dir, title in [
        (ax1, VL0_VIENNA, 'Vienna'),
        (ax2, VL0_CONTRA, 'Contrafold'),
    ]:
        ax.set_facecolor('white')
        for i, model in enumerate(MODEL_ORDER):
            path = base_dir / f'detailed_results_{model}.csv'
            data = load_mean_f1_per_alpha(path)
            if not data:
                continue
            alphas = sorted(data.keys())
            f1s = [data[a] for a in alphas]
            ax.plot(alphas, f1s, 'o-', label=MODEL_LABELS.get(model, model),
                    color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=2, markersize=3, markevery=5)
        ax.set_xlabel('α')
        ax.set_ylabel('Mean F1')
        ax.set_title(f'VL0 — {title}')
        ax.legend(loc='upper right', ncol=2)
        ax.set_xlim(0, 2.05)
        ax.set_ylim(0.5, 0.75)
        ax.grid(alpha=0.3)

    plt.suptitle('Validation (VL0): F1 vs α (0→2, step 0.02)', fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = FIG_DIR / 'vl0_alpha_sweep_both_lightorange.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=LIGHT_ORANGE)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_vl0_vienna()
    fig_vl0_contrafold()
    fig_vl0_both()
    fig_vl0_both_lightorange()
    print("[INFO] VL0 alpha sweep figures saved to", FIG_DIR)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
