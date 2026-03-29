#!/usr/bin/env python3
"""TS0/NEW F1 boxplots by length bin for best config per model."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / 'figures' / 'main' / 'length_boxplot'
FIG_DIR.mkdir(parents=True, exist_ok=True)
BEST_CONFIG_PATH = REPO_ROOT / 'configs' / 'best_config_val_f1.csv'

# [lo, hi): 100-200 excludes 200; 200-400 excludes 400. Labels: non-overlapping.
LEN_BINS = [('<100', 0, 100), ('100–199', 100, 200), ('200–399', 200, 400), ('400+', 400, 999999)]
MODEL_ORDER = ['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert']
MODEL_LABELS = {'ernie': 'ERNIE', 'roberta': 'RoBERTa', 'rnafm': 'RNAFM',
                'rinalmo': 'RiNALMo', 'onehot': 'One-hot', 'rnabert': 'RNABERT'}
COLORS = ['#3498db', '#95a5a6', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
# RGB(255, 245, 242)
LIGHT_ORANGE = (255/255, 245/255, 242/255)


def _per_seq_paths():
    ts = REPO_ROOT / 'results' / 'per_sequence' / 'ts_per_sequence_metrics.csv'
    new = REPO_ROOT / 'results' / 'per_sequence' / 'new_per_sequence_metrics.csv'
    return ts, new


def get_len_bin(length):
    for label, lo, hi in LEN_BINS:
        if lo <= length < hi:
            return label
    return '400+'


def load_best_config():
    cfg = {}
    with open(BEST_CONFIG_PATH) as f:
        for r in csv.DictReader(f):
            m = r['model']
            if m == 'model':
                continue
            cfg[m] = {'layer': int(r['selected_layer']), 'k': int(r['selected_k']),
                      'threshold': float(r['selected_best_threshold']),
                      'decoding_mode': r['selected_decoding_mode']}
    return cfg


def load_partition(path, best_cfg):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            m = r.get('model', '')
            if m not in best_cfg:
                continue
            bc = best_cfg[m]
            if (int(r.get('layer', -1)) != bc['layer'] or int(r.get('k', -1)) != bc['k'] or
                    abs(float(r.get('threshold', -1)) - bc['threshold']) > 1e-6 or
                    r.get('decoding_mode', '') != bc['decoding_mode']):
                continue
            rows.append({'model': m, 'length': int(r.get('length', 0)), 'f1': float(r.get('f1', 0) or 0),
                         'len_bin': get_len_bin(int(r.get('length', 0)))})
    return rows


def plot_boxplot(ax, data, title):
    bin_order = [b[0] for b in LEN_BINS]
    n_bins = len(bin_order)
    n_models = len(MODEL_ORDER)
    width = 0.12
    positions = np.arange(n_bins)
    for i, model in enumerate(MODEL_ORDER):
        model_data = [r for r in data if r['model'] == model]
        if not model_data:
            continue
        vals_by_bin = {b: [] for b in bin_order}
        for r in model_data:
            vals_by_bin[r['len_bin']].append(r['f1'])
        offset = (i - n_models / 2 + 0.5) * width
        box_data = [vals_by_bin[b] if vals_by_bin[b] else [np.nan] for b in bin_order]
        bp = ax.boxplot(box_data, positions=positions + offset, widths=width * 0.8,
                        patch_artist=True, showfliers=False, zorder=1)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS[i])
            patch.set_alpha(0.7)
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(1.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(bin_order)
    ax.set_xlabel('Sequence Length (nt)')
    ax.set_ylabel('F1')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.6, color='#000000', linewidth=1)
    ax.tick_params(axis='both', colors='#000000')
    for spine in ax.spines.values():
        spine.set_color('#000000')
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COLORS[i], alpha=0.7, label=MODEL_LABELS.get(m, m))
                      for i, m in enumerate(MODEL_ORDER)], loc='upper right', ncol=2)


def main():
    ts_path, new_path = _per_seq_paths()
    if not ts_path.exists() or not new_path.exists():
        print("Per-sequence metrics not found.")
        return 1
    best_cfg = load_best_config()
    ts_data = load_partition(ts_path, best_cfg)
    new_data = load_partition(new_path, best_cfg)
    if not ts_data and not new_data:
        print("No per-sequence data matching best config.")
        return 1

    for facecolor, out_name in [('white', 'f1_by_length_boxplot.png'),
                                 (LIGHT_ORANGE, 'f1_by_length_boxplot_lightorange.png')]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(facecolor)
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')
        plot_boxplot(ax1, ts_data, '(a) In-distribution (TEST)')
        plot_boxplot(ax2, new_data, '(b) Out-of-distribution (NEW)')
        plt.tight_layout()
        out_path = FIG_DIR / out_name
        plt.savefig(out_path, facecolor=facecolor)
        plt.close()
        print(f"Saved {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
