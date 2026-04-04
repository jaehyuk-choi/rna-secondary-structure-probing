#!/usr/bin/env python3
"""Quick α=0 vs best-α mean F1 from detailed_results CSVs (no per-seq tests)."""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

def load_detailed(path):
    if not Path(path).exists(): return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({'alpha': float(r['alpha']), 'f1': float(r['f1'])})
    return rows

def mean_f1_at_alpha(data, alpha, tol=0.01):
    if not data: return float('nan')
    alphas = sorted(set(r['alpha'] for r in data))
    closest = min(alphas, key=lambda a: abs(a - alpha))
    vals = [r['f1'] for r in data if abs(r['alpha'] - closest) < tol]
    return sum(vals)/len(vals) if vals else float('nan')

VAL_OPT = {
    ('ernie','Vienna'):1.92, ('ernie','Contrafold'):1.96,
    ('roberta','Vienna'):0.44, ('roberta','Contrafold'):0.56,
    ('rnafm','Vienna'):1.78, ('rnafm','Contrafold'):1.06,
    ('rinalmo','Vienna'):0.00, ('rinalmo','Contrafold'):0.42,
    ('onehot','Vienna'):0.04, ('onehot','Contrafold'):0.74,
    ('rnabert','Vienna'):0.14, ('rnabert','Contrafold'):0.56,
}

FOLDING_RESULTS = REPO_ROOT / 'results' / 'folding'
BASELINE_DIRS = {
    'ts0_vienna': REPO_ROOT / 'results' / 'sweeps' / 'results_ts0',
    'ts0_contrafold': REPO_ROOT / 'results' / 'sweeps' / 'results_ts0_contrafold',
    'new_vienna': REPO_ROOT / 'results' / 'sweeps' / 'results_new',
    'new_contrafold': REPO_ROOT / 'results' / 'sweeps' / 'results_new_contrafold',
}
models = ['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert']
partitions = [
    ('ts0_vienna', 'TS0 Vienna'),
    ('ts0_contrafold', 'TS0 Contrafold'),
    ('new_vienna', 'NEW Vienna'),
    ('new_contrafold', 'NEW Contrafold'),
]

for pkey, pname in partitions:
    print(f"\n{'='*65}")
    print(pname)
    print('='*65)
    print(f"{'model':<10} {'mean F1 (α=0)':<16} {'mean F1 (best α)':<18} {'best α'}")
    print('-'*65)
    backend = 'Vienna' if 'vienna' in pkey else 'Contrafold'
    baseline_dir = BASELINE_DIRS.get(pkey)
    for m in models:
        d25 = load_detailed(FOLDING_RESULTS / pkey / f'detailed_results_{m}.csv')
        d8 = load_detailed(baseline_dir / f'detailed_results_{m}.csv') if baseline_dir else []
        opt_alpha = VAL_OPT.get((m, backend))
        f1_opt = mean_f1_at_alpha(d25, opt_alpha) if d25 and opt_alpha is not None else float('nan')
        f1_0 = mean_f1_at_alpha(d8, 0.0) if d8 else float('nan')
        f0_str = f"{f1_0:.4f}" if f1_0 == f1_0 else "-"
        fopt_str = f"{f1_opt:.4f}" if f1_opt == f1_opt else "-"
        a_str = f"{opt_alpha:.2f}" if opt_alpha is not None else "-"
        print(f"{m:<10} {f0_str:<16} {fopt_str:<18} {a_str}")
