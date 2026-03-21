#!/usr/bin/env python3
"""
Probe Best vs Unconstrained: grouped bar chart (TS0/NEW).
Output: fig1_grouped_bar_f1_lightorange.png — light orange bg, no caption.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MARCH1 = Path(__file__).resolve().parents[1]
DATA_PATH = MARCH1 / 'probe_unconstrained_vs_best_comparison.csv'
OUT_DIR = MARCH1 / 'figures' / 'probe_comparison'
OUT_PATH = OUT_DIR / 'fig1_grouped_bar_f1_lightorange.png'

MODEL_ORDER = ['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert']
MODEL_LABELS = {'ernie': 'ERNIE', 'roberta': 'RoBERTa', 'rnafm': 'RNAFM',
                'rinalmo': 'RiNALMo', 'onehot': 'One-hot', 'rnabert': 'RNABERT'}

# RGB(255, 245, 242)
LIGHT_ORANGE = (255/255, 245/255, 242/255)

# Bar colors: Best=teal, Unconstrained=orange
COLOR_BEST = '#17a2b8'
COLOR_UNCONSTRAINED = '#fd7e14'


def load_data():
    rows = []
    with open(DATA_PATH) as f:
        for r in csv.DictReader(f):
            if not r.get('model'):
                continue
            rows.append({
                'model': r['model'],
                'ts0_best': float(r.get('ts0_best', 0)),
                'ts0_unconstrained': float(r.get('ts0_unconstrained', 0)),
                'new_best': float(r.get('new_best', 0)),
                'new_unconstrained': float(r.get('new_unconstrained', 0)),
            })
    return [r for r in rows if r['model'] in MODEL_ORDER]


def main():
    rows = load_data()
    if not rows:
        print(f"[ERROR] No data in {DATA_PATH}")
        return 1

    models = [r['model'] for r in rows]
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.bar(x - 1.5*width, [r['ts0_best'] for r in rows], width, label='Best (TS0)',
           color=COLOR_BEST, edgecolor='black', linewidth=0.5)
    ax.bar(x - 0.5*width, [r['ts0_unconstrained'] for r in rows], width, label='Unconstrained (TS0)',
           color=COLOR_UNCONSTRAINED, edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.5*width, [r['new_best'] for r in rows], width, label='Best (NEW)',
           color=COLOR_BEST, hatch='///', edgecolor='black', linewidth=0.5)
    ax.bar(x + 1.5*width, [r['new_unconstrained'] for r in rows], width, label='Unconstrained (NEW)',
           color=COLOR_UNCONSTRAINED, hatch='///', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=0)
    ax.set_ylabel('F1')
    ax.set_ylim(0, 0.6)
    ax.legend(loc='upper right', ncol=2, prop=dict(weight='bold'))
    ax.grid(axis='y', alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[INFO] Saved {OUT_PATH}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
