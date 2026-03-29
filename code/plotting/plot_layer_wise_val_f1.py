#!/usr/bin/env python3
"""Val F1 vs layer from layer_wise_val_f1.csv."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / 'results' / 'sweeps' / 'layer_wise_val_f1.csv'
OUT_DIR = REPO_ROOT / 'figures' / 'main' / 'layer_k_comparison'

# Original colors (like k_comparison)
MODEL_COLORS = {
    'ERNIE': '#e74c3c', 'RoBERTa': '#3498db', 'RNAFM': '#2ecc71',
    'RiNALMo': '#9b59b6', 'One-hot': '#f39c12', 'RNABERT': '#1abc9c',
}
MODEL_MARKERS = ['o', 's', '^', 'D', 'v', 'p']

# RGB(255, 245, 242)
LIGHT_ORANGE = (255/255, 245/255, 242/255)


def load_data():
    rows = []
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        layer_cols = [c for c in reader.fieldnames if c.startswith('L') and len(c) > 1 and c[1:].isdigit()]
        for r in reader:
            model = r.get('model', '')
            if not model:
                continue
            vals = []
            layers = []
            for i, col in enumerate(layer_cols):
                v = r.get(col, '').strip()
                if v:
                    try:
                        vals.append(float(v))
                        layers.append(int(col[1:]))
                    except ValueError:
                        pass
            if vals:
                rows.append({'model': model, 'layers': layers, 'f1': vals})
    return rows


def plot_layer_wise(rows, facecolor='white', out_name='layer_wise_val_f1.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor('white')

    for i, row in enumerate(rows):
        model = row['model']
        color = MODEL_COLORS.get(model, f'C{i}')
        marker = MODEL_MARKERS[i % len(MODEL_MARKERS)]
        # markevery=1: marker at every point so last point is consistent (no orphan)
        ax.plot(row['layers'], row['f1'], '-', label=model, color=color,
                marker=marker, markersize=3, linewidth=2, markevery=1)

    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('Val F1')
    ax.set_xlim(-0.5, 33.5)
    ax.set_ylim(0, 0.65)
    ax.legend(loc='upper right', fontsize=14, prop=dict(weight='bold'))
    ax.grid(alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=facecolor)
    plt.close()
    print(f"Saved {out_path}")


def main():
    rows = load_data()
    if not rows:
        print(f"error: No data in {DATA_PATH}")
        return 1

    # Original: white background
    plot_layer_wise(rows, facecolor='white', out_name='layer_wise_val_f1.png')
    # New file: light orange background
    plot_layer_wise(rows, facecolor=LIGHT_ORANGE, out_name='layer_wise_val_f1_lightorange.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
