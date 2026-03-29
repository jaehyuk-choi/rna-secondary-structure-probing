#!/usr/bin/env python3
"""Probe-only TS0 vs NEW F1 and precision–recall from unconstrained_results_summary (or final_* fallback)."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / 'results' / 'metrics' / 'unconstrained_results_summary.csv'
FINAL_TS0 = REPO_ROOT / 'results' / 'metrics' / 'final_test_metrics.csv'
FINAL_NEW = REPO_ROOT / 'results' / 'metrics' / 'final_new_metrics.csv'
FIG_DIR = REPO_ROOT / 'figures' / 'main'

MODEL_LABELS = {'ernie': 'ERNIE', 'roberta': 'RoBERTa', 'rnafm': 'RNAFM',
                'rinalmo': 'RiNALMo', 'onehot': 'One-hot', 'rnabert': 'RNABERT'}


def load_data():
    if not DATA_PATH.exists():
        return []
    rows = []
    with open(DATA_PATH) as f:
        for r in csv.DictReader(f):
            if not r.get('model'):
                continue
            for k in ['ts0_f1', 'ts0_precision', 'ts0_recall', 'new_f1', 'new_precision', 'new_recall']:
                try:
                    val = r.get(k, '').strip()
                    r[k] = float(val) if val else np.nan
                except (ValueError, TypeError):
                    r[k] = np.nan
            rows.append(r)
    return rows


def fig_probe_f1(rows):
    """TS0 vs NEW F1 by model."""
    rows = [r for r in rows if not np.isnan(r.get('ts0_f1', np.nan)) and not np.isnan(r.get('new_f1', np.nan))]
    if not rows:
        print("warn: No valid probe F1 data")
        return

    models = [r['model'] for r in rows]
    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, [r['ts0_f1'] for r in rows], w, label='TS0', color='#3498db')
    ax.bar(x + w/2, [r['new_f1'] for r in rows], w, label='NEW', color='#e74c3c')

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=30, ha='right')
    ax.set_ylabel('F1')
    ax.set_title('Probe-only F1: TS0 vs NEW (unconstrained)')
    ax.legend()
    ax.set_ylim(0, 0.6)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'probe_f1_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {FIG_DIR / 'probe_f1_comparison.png'}")


def fig_precision_recall(rows):
    """Precision vs Recall scatter by model."""
    rows = [r for r in rows if not np.isnan(r.get('ts0_precision', np.nan)) and not np.isnan(r.get('ts0_recall', np.nan))]
    if not rows:
        print("warn: No valid precision/recall data")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        ax.scatter(r['ts0_recall'], r['ts0_precision'], s=120, label=MODEL_LABELS.get(r['model'], r['model']))

    ax.set_xlabel('Recall (TS0)')
    ax.set_ylabel('Precision (TS0)')
    ax.set_title('Probe-only: Precision vs Recall (TS0, unconstrained)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'probe_precision_recall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {FIG_DIR / 'probe_precision_recall.png'}")


def load_from_final_metrics():
    """Fallback: merge final_test_metrics and final_new_metrics."""
    rows = []
    ts0 = {}
    if FINAL_TS0.exists():
        with open(FINAL_TS0) as f:
            for r in csv.DictReader(f):
                ts0[r['model']] = {'ts0_f1': float(r.get('f1', 0)), 'ts0_precision': float(r.get('precision', 0)),
                                    'ts0_recall': float(r.get('recall', 0))}
    new = {}
    if FINAL_NEW.exists():
        with open(FINAL_NEW) as f:
            for r in csv.DictReader(f):
                new[r['model']] = {'new_f1': float(r.get('f1', 0)), 'new_precision': float(r.get('precision', 0)),
                                    'new_recall': float(r.get('recall', 0))}
    for m in set(ts0.keys()) | set(new.keys()):
        r = {'model': m}
        r.update(ts0.get(m, {'ts0_f1': np.nan, 'ts0_precision': np.nan, 'ts0_recall': np.nan}))
        r.update(new.get(m, {'new_f1': np.nan, 'new_precision': np.nan, 'new_recall': np.nan}))
        rows.append(r)
    return rows


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    valid = [r for r in rows if not (np.isnan(r.get('ts0_f1', np.nan)) or np.isnan(r.get('new_f1', np.nan)))]
    if len(valid) < 2:
        rows = load_from_final_metrics()
    if not rows:
        print("error: No data loaded")
        return 1
    fig_probe_f1(rows)
    fig_precision_recall(rows)
    print("figures ->", FIG_DIR)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
