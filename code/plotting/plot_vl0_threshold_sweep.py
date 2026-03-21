#!/usr/bin/env python3
"""
Validation (VL0): F1 vs decoding threshold τ (0.5–0.95).
Line plot per model, best config (layer, k, decoding_mode).
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MARCH1 = Path(__file__).resolve().parents[1]
FIG_DIR = MARCH1 / 'figures'
FEB8 = Path('/projects/u6cg/jay/dissertations/feb8/results_updated')
BEST_CONFIG = FEB8 / 'summary/final_selected_config.csv'
OUTPUTS = FEB8 / 'outputs'

MODEL_ORDER = ['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert']
MODEL_LABELS = {'ernie': 'ERNIE', 'roberta': 'RoBERTa', 'rnafm': 'RNAFM',
                'rinalmo': 'RiNALMo', 'onehot': 'One-hot', 'rnabert': 'RNABERT'}
LINE_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

LIGHT_ORANGE = (255/255, 245/255, 242/255)


def load_best_config():
    cfg = {}
    with open(BEST_CONFIG) as f:
        for r in csv.DictReader(f):
            m = r['model']
            if m == 'model':
                continue
            cfg[m] = {
                'layer': int(r['selected_layer']),
                'k': int(r['selected_k']),
                'threshold': float(r['selected_best_threshold']),
                'decoding_mode': r['selected_decoding_mode']
            }
    return cfg


def sweep_path(model, best_cfg):
    layer, k, mode = best_cfg['layer'], best_cfg['k'], best_cfg['decoding_mode']
    base = OUTPUTS / model / f'layer_{layer}' / f'k_{k}' / 'seed_42'
    if model == 'rnabert':
        return base / 'val_threshold_sweep.csv'
    return base / f'val_threshold_sweep_{mode}.csv'


def load_sweep_data(path):
    """Return (thresholds, f1s) sorted by threshold."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            t = float(r['threshold'])
            f1 = float(r['f1']) if r.get('f1') and r['f1'] != 'nan' else 0
            rows.append((t, f1))
    rows.sort(key=lambda x: x[0])
    return [x[0] for x in rows], [x[1] for x in rows]


def main():
    best_cfg = load_best_config()

    for facecolor, out_name in [('white', 'vl0_threshold_sweep.png'),
                                (LIGHT_ORANGE, 'vl0_threshold_sweep_lightorange.png')]:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(facecolor)
        ax.set_facecolor('white')

        for i, model in enumerate(MODEL_ORDER):
            if model not in best_cfg:
                continue
            p = sweep_path(model, best_cfg[model])
            if not p.exists():
                print(f"[WARN] Missing {p}")
                continue
            threshs, f1s = load_sweep_data(p)
            tau_star = best_cfg[model]['threshold']
            ax.plot(threshs, f1s, '-', label=f"{MODEL_LABELS.get(model, model)} ({tau_star:.2f})",
                    color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=2)
            # Mark selected τ* with star
            idx = min(range(len(threshs)), key=lambda j: abs(threshs[j] - tau_star))
            ax.scatter([threshs[idx]], [f1s[idx]], marker='*', s=300,
                       color=LINE_COLORS[i % len(LINE_COLORS)], edgecolors='black',
                       linewidths=0.5, zorder=5)

        ax.set_xlabel('Decoding threshold τ')
        ax.set_ylabel('Validation F1 (VL0)')
        ax.set_title('Validation F1 vs decoding threshold τ (0.5–0.95)')
        ax.legend(loc='upper left', ncol=2, prop=dict(weight='bold'))
        ax.set_xlim(0.48, 0.97)
        ax.set_ylim(0, 0.65)
        ax.grid(alpha=0.3)

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        out_path = FIG_DIR / out_name
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=facecolor)
        plt.close()
        print(f"[INFO] Saved {out_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
