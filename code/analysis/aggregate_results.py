#!/usr/bin/env python3
"""
Aggregate CPLfold results into summary CSVs.

For each model × (Vienna/Contrafold) × (TS0/NEW):
- F1 at optimal alpha (from Val if --val-results-dir provided, else from alpha sweep)
- Writes: feb25/results_thresholded_ts0_new/summary.csv and detailed CSVs
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_detailed(base_dir, model):
    """Load detailed_results_{model}.csv, return list of dicts."""
    p = Path(base_dir) / f'detailed_results_{model}.csv'
    if not p.exists():
        return None
    rows = []
    with open(p) as f:
        r = csv.DictReader(f)
        for row in r:
            row['alpha'] = float(row['alpha'])
            row['f1'] = float(row['f1'])
            rows.append(row)
    return rows


def find_optimal_alpha(data):
    """Global optimal: alpha that maximizes mean F1."""
    if not data:
        return None, None
    by_alpha = defaultdict(list)
    for r in data:
        by_alpha[r['alpha']].append(r['f1'])
    alphas = sorted(by_alpha.keys())
    best_alpha = max(alphas, key=lambda a: np.mean(by_alpha[a]))
    best_f1 = np.mean(by_alpha[best_alpha])
    return best_alpha, best_f1


def get_f1_at_alpha(data, alpha, tol=0.01):
    """Get mean F1 at given alpha (or closest available)."""
    if not data:
        return np.nan
    alphas = sorted(set(r['alpha'] for r in data))
    closest = min(alphas, key=lambda a: abs(a - alpha))
    vals = [r['f1'] for r in data if abs(r['alpha'] - closest) < tol]
    return np.mean(vals) if vals else np.nan


def aggregate_alpha_sweep(data):
    """Return dict: alpha -> mean_f1."""
    if not data:
        return {}
    by_alpha = defaultdict(list)
    for r in data:
        by_alpha[r['alpha']].append(r['f1'])
    return {a: np.mean(vals) for a, vals in sorted(by_alpha.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='/projects/u6cg/jay/dissertations/feb25/results_thresholded_ts0_new')
    ap.add_argument('--val-results-dir', default=None,
                    help='VL0 Vienna results dir for optimal alpha. If None, use best from sweep.')
    ap.add_argument('--val-contrafold-dir', default=None,
                    help='VL0 Contrafold results dir for optimal alpha.')
    ap.add_argument('--output-dir', default=None,
                    help='Same as results-dir if not specified')
    ap.add_argument('--models', nargs='+', default=['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot'])
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    subdirs = {
        'ts0_vienna': ('TS0', 'Vienna'),
        'ts0_contrafold': ('TS0', 'Contrafold'),
        'new_vienna': ('NEW', 'Vienna'),
        'new_contrafold': ('NEW', 'Contrafold'),
    }

    # Optional: load Val optimal alphas
    val_opt_alpha = {}  # (model, backend) -> alpha
    if args.val_results_dir:
        val_vienna = Path(args.val_results_dir)
        val_contra = Path(args.val_contrafold_dir) if args.val_contrafold_dir else None
        for m in args.models:
            d = load_detailed(val_vienna, m)
            if d:
                a, _ = find_optimal_alpha(d)
                val_opt_alpha[(m, 'Vienna')] = a
            if val_contra:
                d = load_detailed(val_contra, m)
                if d:
                    a, _ = find_optimal_alpha(d)
                    val_opt_alpha[(m, 'Contrafold')] = a

    summary_rows = []
    for subdir, (partition, backend) in subdirs.items():
        base = results_dir / subdir
        if not base.exists():
            continue
        for model in args.models:
            data = load_detailed(base, model)
            if not data:
                continue
            # Optimal alpha: from Val if available, else from sweep
            opt_alpha = val_opt_alpha.get((model, backend))
            if opt_alpha is None:
                opt_alpha, _ = find_optimal_alpha(data)
            f1_at_opt = get_f1_at_alpha(data, opt_alpha) if opt_alpha is not None else np.nan
            alpha_source = 'Val' if (model, backend) in val_opt_alpha else 'sweep'
            summary_rows.append({
                'model': model,
                'partition': partition,
                'backend': backend,
                'optimal_alpha': opt_alpha if opt_alpha is not None else '',
                'alpha_source': alpha_source,
                'mean_f1': f1_at_opt,
            })

    # Write summary.csv
    summary_path = out_dir / 'summary.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'partition', 'backend', 'optimal_alpha', 'alpha_source', 'mean_f1'])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[INFO] Wrote {summary_path}")

    # Write detailed alpha sweep per (partition, backend)
    for subdir, (partition, backend) in subdirs.items():
        base = results_dir / subdir
        if not base.exists():
            continue
        rows = []
        for model in args.models:
            data = load_detailed(base, model)
            if not data:
                continue
            sweep = aggregate_alpha_sweep(data)
            for alpha, mean_f1 in sweep.items():
                rows.append({'model': model, 'alpha': alpha, 'mean_f1': mean_f1})
        if rows:
            detail_path = out_dir / f'detailed_alpha_sweep_{subdir}.csv'
            with open(detail_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['model', 'alpha', 'mean_f1'])
                w.writeheader()
                w.writerows(rows)
            print(f"[INFO] Wrote {detail_path}")

    print(f"[INFO] Done. Summary: {summary_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
