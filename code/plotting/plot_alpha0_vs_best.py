#!/usr/bin/env python3
"""
Visualize α=0 vs best α comparison (CPLfold results).

Generates:
  1. fig1_grouped_bar.png - mean F1: α=0 vs best α by model × partition
  2. fig2_pct_improvement_heatmap.png - %Δ improvement heatmap
  3. fig3_best_alpha.png - optimal α per model × backend
  4. fig4_significance_summary.png - significance indicators
  5. fig5_combined_panel.png - 2×2 summary panel
  6. fig6_line_vienna.png - line chart: TS0→NEW F1 by model (Vienna)
  7. fig7_line_contrafold.png - line chart: TS0→NEW F1 by model (Contrafold)
  8. fig8_line_both.png - both backends in 2 subplots
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MARCH1 = Path(__file__).resolve().parents[1]
DATA_PATH = MARCH1 / 'data' / 'alpha0_vs_best_full.csv'
FIG_DIR = MARCH1 / 'figures'

# Model order for consistent display
MODEL_ORDER = ['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert']
MODEL_LABELS = {'ernie': 'ERNIE', 'roberta': 'RoBERTa', 'rnafm': 'RNAFM',
                'rinalmo': 'RiNALMo', 'onehot': 'One-hot', 'rnabert': 'RNABERT'}


def load_data():
    rows = []
    with open(DATA_PATH) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if not r.get('partition'):
                continue
            r['mean_alpha0'] = float(r['mean(α=0)'])
            r['mean_best'] = float(r['mean(best)'])
            r['best_alpha'] = float(r['best_α']) if r['best_α'] != 'nan' else 0.0
            pct = r['%Δ'].replace('+', '').replace('%', '')
            r['pct_delta'] = float(pct) if pct else 0.0
            r['p_ttest'] = r['p_ttest']
            r['sig'] = '***' if r['p<0.001'] == '✓' else '**' if r['p<0.005'] == '✓' else '*' if r['p<0.01'] == '✓' else 'n.s.'
            rows.append(r)
    return rows


def fig1_grouped_bar(rows):
    """Grouped bar: mean(α=0) vs mean(best) by model, faceted by partition."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    axes = axes.flatten()

    partitions = ['TS0 Vienna', 'TS0 Contrafold', 'NEW Vienna', 'NEW Contrafold']
    colors = {'α=0': '#95a5a6', 'best α': '#3498db'}

    for ax, part in zip(axes, partitions):
        sub = [r for r in rows if r['partition'] == part]
        models = [r['model'] for r in sub]
        x = np.arange(len(models))
        w = 0.35

        mean0 = [r['mean_alpha0'] for r in sub]
        mean_best = [r['mean_best'] for r in sub]

        ax.bar(x - w/2, mean0, w, label='α=0', color=colors['α=0'])
        ax.bar(x + w/2, mean_best, w, label='best α', color=colors['best α'])

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=30, ha='right')
        ax.set_ylabel('Mean F1')
        ax.set_title(part)
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0, 0.85)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig1_grouped_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {FIG_DIR / 'fig1_grouped_bar.png'}")


def fig2_heatmap(rows):
    """%Δ improvement heatmap: model × partition."""
    models = MODEL_ORDER
    parts = ['TS0 Vienna', 'TS0 Contrafold', 'NEW Vienna', 'NEW Contrafold']
    part_short = ['TS0-V', 'TS0-C', 'NEW-V', 'NEW-C']

    mat = np.full((len(models), len(parts)), np.nan)
    for r in rows:
        mi = models.index(r['model']) if r['model'] in models else -1
        pi = parts.index(r['partition']) if r['partition'] in parts else -1
        if mi >= 0 and pi >= 0:
            mat[mi, pi] = r['pct_delta']

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=25)

    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels(part_short)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models])

    for i in range(len(models)):
        for j in range(len(parts)):
            v = mat[i, j]
            txt = f'{v:+.1f}%' if not np.isnan(v) else '-'
            ax.text(j, i, txt, ha='center', va='center', fontsize=10)

    plt.colorbar(im, ax=ax, label='% Δ F1')
    ax.set_title('F1 Improvement: best α vs α=0')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig2_pct_improvement_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {FIG_DIR / 'fig2_pct_improvement_heatmap.png'}")


def fig3_best_alpha(rows):
    """Optimal α per model × backend (Vienna vs Contrafold)."""
    vienna = [r for r in rows if 'Vienna' in r['partition']]
    contra = [r for r in rows if 'Contrafold' in r['partition']]

    # Use TS0 as representative
    vienna = [r for r in vienna if r['partition'] == 'TS0 Vienna']
    contra = [r for r in contra if r['partition'] == 'TS0 Contrafold']

    models = [r['model'] for r in vienna]
    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, [r['best_alpha'] for r in vienna], w, label='Vienna', color='#3498db')
    ax.bar(x + w/2, [r['best_alpha'] for r in contra], w, label='Contrafold', color='#e74c3c')

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=30, ha='right')
    ax.set_ylabel('Optimal α')
    ax.set_title('Val-optimal α by model and folding backend')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig3_best_alpha.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {FIG_DIR / 'fig3_best_alpha.png'}")


def fig4_significance(rows):
    """Significance summary: which models show ***/**/*."""
    models = MODEL_ORDER
    parts = ['TS0 Vienna', 'TS0 Contrafold', 'NEW Vienna', 'NEW Contrafold']
    part_short = ['TS0-V', 'TS0-C', 'NEW-V', 'NEW-C']

    sig_map = {'***': 3, '**': 2, '*': 1, 'n.s.': 0}
    mat = np.zeros((len(models), len(parts)))
    for r in rows:
        mi = models.index(r['model']) if r['model'] in models else -1
        pi = parts.index(r['partition']) if r['partition'] in parts else -1
        if mi >= 0 and pi >= 0:
            mat[mi, pi] = sig_map.get(r['sig'], 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        cmap = plt.colormaps['YlOrRd'].resampled(4)
    except AttributeError:
        cmap = plt.cm.get_cmap('YlOrRd', 4)
    im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=3)

    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels(part_short)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models])

    for i in range(len(models)):
        for j in range(len(parts)):
            s = 'n.s.'
            for r in rows:
                if r['model'] == models[i] and r['partition'] == parts[j]:
                    s = r['sig']
                    break
            ax.text(j, i, s, ha='center', va='center', fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['n.s.', 'p<0.01', 'p<0.005', 'p<0.001'])
    ax.set_title('Statistical significance (paired t-test)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4_significance_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {FIG_DIR / 'fig4_significance_summary.png'}")


def fig5_combined(rows):
    """2×2 combined panel."""
    fig = plt.figure(figsize=(14, 12))

    # (1) Grouped bar - TS0 Vienna only
    ax1 = fig.add_subplot(2, 2, 1)
    sub = [r for r in rows if r['partition'] == 'TS0 Vienna']
    models = [r['model'] for r in sub]
    x = np.arange(len(models))
    w = 0.35
    ax1.bar(x - w/2, [r['mean_alpha0'] for r in sub], w, label='α=0', color='#95a5a6')
    ax1.bar(x + w/2, [r['mean_best'] for r in sub], w, label='best α', color='#3498db')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=30, ha='right')
    ax1.set_ylabel('Mean F1')
    ax1.set_title('TS0 Vienna: α=0 vs best α')
    ax1.legend()
    ax1.set_ylim(0, 0.85)

    # (2) Grouped bar - NEW Vienna only
    ax2 = fig.add_subplot(2, 2, 2)
    sub = [r for r in rows if r['partition'] == 'NEW Vienna']
    models = [r['model'] for r in sub]
    x = np.arange(len(models))
    ax2.bar(x - w/2, [r['mean_alpha0'] for r in sub], w, label='α=0', color='#95a5a6')
    ax2.bar(x + w/2, [r['mean_best'] for r in sub], w, label='best α', color='#3498db')
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=30, ha='right')
    ax2.set_ylabel('Mean F1')
    ax2.set_title('NEW Vienna: α=0 vs best α')
    ax2.legend()
    ax2.set_ylim(0, 0.85)

    # (3) %Δ heatmap
    ax3 = fig.add_subplot(2, 2, 3)
    models = MODEL_ORDER
    parts = ['TS0 Vienna', 'TS0 Contrafold', 'NEW Vienna', 'NEW Contrafold']
    part_short = ['TS0-V', 'TS0-C', 'NEW-V', 'NEW-C']
    mat = np.full((len(models), len(parts)), np.nan)
    for r in rows:
        mi = models.index(r['model']) if r['model'] in models else -1
        pi = parts.index(r['partition']) if r['partition'] in parts else -1
        if mi >= 0 and pi >= 0:
            mat[mi, pi] = r['pct_delta']
    im = ax3.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=25)
    ax3.set_xticks(range(len(parts)))
    ax3.set_xticklabels(part_short)
    ax3.set_yticks(range(len(models)))
    ax3.set_yticklabels([MODEL_LABELS.get(m, m) for m in models])
    for i in range(len(models)):
        for j in range(len(parts)):
            v = mat[i, j]
            txt = f'{v:+.1f}%' if not np.isnan(v) else '-'
            ax3.text(j, i, txt, ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax3, label='% Δ')
    ax3.set_title('% F1 improvement')

    # (4) Best α
    ax4 = fig.add_subplot(2, 2, 4)
    vienna = [r for r in rows if r['partition'] == 'TS0 Vienna']
    contra = [r for r in rows if r['partition'] == 'TS0 Contrafold']
    models = [r['model'] for r in vienna]
    x = np.arange(len(models))
    ax4.bar(x - w/2, [r['best_alpha'] for r in vienna], w, label='Vienna', color='#3498db')
    ax4.bar(x + w/2, [r['best_alpha'] for r in contra], w, label='Contrafold', color='#e74c3c')
    ax4.set_xticks(x)
    ax4.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=30, ha='right')
    ax4.set_ylabel('Optimal α')
    ax4.set_title('Val-optimal α')
    ax4.legend()

    plt.suptitle('CPLfold: α=0 vs best α (Val-optimal)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig5_combined_panel.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {FIG_DIR / 'fig5_combined_panel.png'}")


# Line chart colors (distinct per model)
LINE_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']


def fig6_line_vienna(rows):
    """Line chart: TS0 → NEW mean F1 by model (Vienna)."""
    vienna = [r for r in rows if 'Vienna' in r['partition']]
    x_labels = ['TS0', 'NEW']
    x_pos = [0, 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, m in enumerate(MODEL_ORDER):
        pts = [r for r in vienna if r['model'] == m]
        pts = sorted(pts, key=lambda r: 0 if 'TS0' in r['partition'] else 1)
        if len(pts) == 2:
            y = [pts[0]['mean_best'], pts[1]['mean_best']]
            ax.plot(x_pos, y, 'o-', label=MODEL_LABELS.get(m, m),
                    color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=2, markersize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Mean F1 (best α)')
    ax.set_title('Vienna: TS0 → NEW F1 by model')
    ax.legend(loc='lower right', ncol=2)
    ax.set_ylim(0.5, 0.8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig6_line_vienna.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {FIG_DIR / 'fig6_line_vienna.png'}")


def fig7_line_contrafold(rows):
    """Line chart: TS0 → NEW mean F1 by model (Contrafold)."""
    contra = [r for r in rows if 'Contrafold' in r['partition']]
    x_labels = ['TS0', 'NEW']
    x_pos = [0, 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, m in enumerate(MODEL_ORDER):
        pts = [r for r in contra if r['model'] == m]
        pts = sorted(pts, key=lambda r: 0 if 'TS0' in r['partition'] else 1)
        if len(pts) == 2:
            y = [pts[0]['mean_best'], pts[1]['mean_best']]
            ax.plot(x_pos, y, 'o-', label=MODEL_LABELS.get(m, m),
                    color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=2, markersize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Mean F1 (best α)')
    ax.set_title('Contrafold: TS0 → NEW F1 by model')
    ax.legend(loc='lower right', ncol=2)
    ax.set_ylim(0.5, 0.8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig7_line_contrafold.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {FIG_DIR / 'fig7_line_contrafold.png'}")


def fig8_line_both(rows):
    """Line chart: Vienna + Contrafold in 2 subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for backend, ax, title in [
        ('Vienna', ax1, 'Vienna'),
        ('Contrafold', ax2, 'Contrafold'),
    ]:
        sub = [r for r in rows if backend in r['partition']]
        x_pos = [0, 1]
        x_labels = ['TS0', 'NEW']
        for i, m in enumerate(MODEL_ORDER):
            pts = [r for r in sub if r['model'] == m]
            pts = sorted(pts, key=lambda r: 0 if 'TS0' in r['partition'] else 1)
            if len(pts) == 2:
                y = [pts[0]['mean_best'], pts[1]['mean_best']]
                ax.plot(x_pos, y, 'o-', label=MODEL_LABELS.get(m, m),
                        color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=2, markersize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('Mean F1 (best α)')
        ax.set_title(title)
        ax.legend(loc='lower right', ncol=2)
        ax.set_ylim(0.5, 0.8)
        ax.grid(alpha=0.3)

    plt.suptitle('CPLfold: TS0 → NEW F1 by model', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig8_line_both.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {FIG_DIR / 'fig8_line_both.png'}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    if not rows:
        print("[ERROR] No data loaded")
        return 1
    fig1_grouped_bar(rows)
    fig2_heatmap(rows)
    fig3_best_alpha(rows)
    fig4_significance(rows)
    fig5_combined(rows)
    fig6_line_vienna(rows)
    fig7_line_contrafold(rows)
    fig8_line_both(rows)
    print("[INFO] All figures saved to", FIG_DIR)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
