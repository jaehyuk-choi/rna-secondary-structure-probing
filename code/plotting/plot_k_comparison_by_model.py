#!/usr/bin/env python3
"""
k comparison: X-axis = model, color = k (32, 64, 128).
Output: k_comparison_val_f1_by_model.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MARCH1 = Path(__file__).resolve().parents[1]
DATA_PATH = MARCH1 / 'data' / 'k_comparison_val_f1.csv'
OUT_DIR = MARCH1 / 'figures' / 'layer_k_comparison'
OUT_PATH = OUT_DIR / 'k_comparison_val_f1_by_model.png'

# NeurIPS-style muted palette (k=32, 64, 128)
K_COLORS = {'32': '#0173B2', '64': '#029E73', '128': '#DE8F05'}


def load_data():
    rows = []
    with open(DATA_PATH) as f:
        for r in csv.DictReader(f):
            rows.append({
                'model': r['model'],
                'k32': float(r.get('k32', 0)),
                'k64': float(r.get('k64', 0)),
                'k128': float(r.get('k128', 0)),
            })
    return rows


def main():
    rows = load_data()
    if not rows:
        print(f"[ERROR] No data in {DATA_PATH}")
        return 1

    models = [r['model'] for r in rows]
    x = np.arange(len(models))
    width = 0.25
    k_keys = ['k32', 'k64', 'k128']
    k_offsets = [-width, 0, width]
    k_labels = ['k=32', 'k=64', 'k=128']

    fig, ax = plt.subplots(figsize=(10, 6))

    for ki, (kkey, koff, klab) in enumerate(zip(k_keys, k_offsets, k_labels)):
        heights = [r[kkey] for r in rows]
        ax.bar(x + koff, heights, width, label=klab, color=K_COLORS[str(32 * (2 ** ki))])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_xlabel('Model')
    ax.set_ylabel('Val F1')
    ax.set_ylim(0, 0.65)
    ax.legend(loc='upper right', fontsize=14, prop=dict(weight='bold'))
    ax.grid(axis='y', alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {OUT_PATH}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
