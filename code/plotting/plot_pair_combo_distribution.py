#!/usr/bin/env python3
"""
Pair combination distribution: One-hot vs RiNALMo.
Bar chart comparing predicted pair type rates (WC+GU highlighted).
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MARCH1 = Path(__file__).resolve().parents[1]
DATA_DIR = MARCH1 / 'data'
FIG_DIR = MARCH1 / 'figures' / 'pair_combo'
OUT_WHITE = FIG_DIR / 'pair_combo_distribution.png'
OUT_LIGHT = FIG_DIR / 'pair_combo_distribution_lightorange.png'

# Pair order (16 combinations)
PAIR_ORDER = ['AA', 'AU', 'AG', 'AC', 'UA', 'UU', 'UG', 'UC', 'GA', 'GU', 'GG', 'GC', 'CA', 'CU', 'CG', 'CC']

# RGB(255, 245, 242)
LIGHT_ORANGE = (255/255, 245/255, 242/255)
COLOR_ONEHOT = '#3498db'
COLOR_RINALMO = '#e74c3c'
# WC+GU pairs for reference
WC_GU_PAIRS = {'AU', 'UA', 'UG', 'GU', 'GC', 'CG'}


def load_model_data(path):
    """Return {pair: rate}."""
    rates = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rates[r['pair']] = float(r['rate'])
    return rates


def main():
    onehot_path = DATA_DIR / 'onehot_pair_combo_distribution.csv'
    rinalmo_path = DATA_DIR / 'rinalmo_pair_combo_distribution.csv'
    if not onehot_path.exists() or not rinalmo_path.exists():
        print("[ERROR] Missing onehot_pair_combo_distribution.csv or rinalmo_pair_combo_distribution.csv")
        return 1

    onehot_rates = load_model_data(onehot_path)
    rinalmo_rates = load_model_data(rinalmo_path)

    x = np.arange(len(PAIR_ORDER))
    width = 0.35

    for facecolor, out_path in [('white', OUT_WHITE), (LIGHT_ORANGE, OUT_LIGHT)]:
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor(facecolor)
        ax.set_facecolor('white')

        onehot_vals = [onehot_rates.get(p, 0) for p in PAIR_ORDER]
        rinalmo_vals = [rinalmo_rates.get(p, 0) for p in PAIR_ORDER]

        ax.bar(x - width/2, onehot_vals, width, label='One-hot', color=COLOR_ONEHOT,
               edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, rinalmo_vals, width, label='RiNALMo', color=COLOR_RINALMO,
               edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(PAIR_ORDER, fontsize=10)
        for i, p in enumerate(PAIR_ORDER):
            if p in WC_GU_PAIRS:
                ax.get_xticklabels()[i].set_fontweight('bold')
                ax.get_xticklabels()[i].set_color('#27ae60')

        ax.set_xlabel('Pair type (bold green = WC+GU canonical)')
        ax.set_ylabel('Rate (fraction of predicted pairs)')
        ax.set_title('Predicted pair type distribution: One-hot vs RiNALMo')
        ax.legend(loc='upper right')
        ax.set_ylim(0, max(max(onehot_vals), max(rinalmo_vals)) * 1.15)
        ax.grid(axis='y', alpha=0.3)

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=facecolor)
        plt.close()
        print(f"[INFO] Saved {out_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
