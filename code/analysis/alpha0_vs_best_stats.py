#!/usr/bin/env python3
"""Paired α=0 vs best-α F1 (t-test / Wilcoxon). RNABERT at α=0 borrows ERNIE seq scores (bonus off)."""
import csv
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def load_seq_level(path, alpha, tol=0.01):
    if not Path(path).exists(): return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            a = float(r['alpha'])
            if abs(a - alpha) < tol:
                out[r['seq_id']] = float(r['f1'])
    return out

def load_seq_level_any(path):
    if not Path(path).exists(): return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r['seq_id']] = float(r['f1'])
    return out

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

# For rnabert α=0: use ernie's α=0 (same folding when bonus=0)
ALPHA0_REF_MODEL = 'ernie'

csv_rows = []

print("="*120)
print("α=0 vs best α (mean F1, %Δ, paired tests)")
print("RNABERT α=0: per-seq F1 from ERNIE at α=0")
print("="*90)

for pkey, pname in partitions:
    print(f"\n### {pname} ###")
    backend = 'Vienna' if 'vienna' in pkey else 'Contrafold'
    baseline_dir = BASELINE_DIRS.get(pkey)

    f1_0_ref = load_seq_level(baseline_dir / f'detailed_results_{ALPHA0_REF_MODEL}.csv', 0.0) if baseline_dir else {}
    mean_alpha0_partition = np.mean(list(f1_0_ref.values())) if f1_0_ref else float('nan')

    print(f"{'partition':<18} {'model':<10} {'mean(α=0)':<10} {'mean(best)':<10} {'best_α':<8} {'%Δ':<10} {'p<0.001':<8} {'p<0.005':<8} {'p<0.01':<8} {'p_ttest':<12} {'p_wilcoxon':<12}")
    print("-"*120)

    for m in models:
        opt_alpha = VAL_OPT.get((m, backend))
        d25 = load_seq_level_any(FOLDING_RESULTS / pkey / f'detailed_results_{m}.csv')
        if m == 'rnabert':
            d0 = f1_0_ref
        else:
            d0 = load_seq_level(baseline_dir / f'detailed_results_{m}.csv', 0.0) if baseline_dir else {}

        common = sorted(set(d0.keys()) & set(d25.keys()))
        if not common:
            print(f"{pname:<18} {m:<10} -          -          -        -          -            -            -")
            continue

        arr_0 = np.array([d0[s] for s in common])
        arr_best = np.array([d25[s] for s in common])
        mean_0 = mean_alpha0_partition if m == 'rnabert' else arr_0.mean()
        mean_best = arr_best.mean()

        pct = 100 * (mean_best - mean_0) / mean_0 if mean_0 > 0 else 0

        if HAS_SCIPY and len(common) >= 3:
            try:
                _, p_ttest = stats.ttest_rel(arr_best, arr_0)
            except Exception:
                p_ttest = np.nan
            try:
                _, p_wilcoxon = stats.wilcoxon(arr_best, arr_0)
            except Exception:
                p_wilcoxon = np.nan
            s001 = "✓" if not np.isnan(p_ttest) and p_ttest < 0.001 else ""
            s005 = "✓" if not np.isnan(p_ttest) and p_ttest < 0.005 else ""
            s01 = "✓" if not np.isnan(p_ttest) and p_ttest < 0.01 else "n.s." if np.isnan(p_ttest) else "n.s."
            if not np.isnan(p_ttest) and p_ttest >= 0.01: s01 = "n.s."
            p_t_str = f"{p_ttest:.2e}" if not np.isnan(p_ttest) else "nan"
            p_w_str = f"{p_wilcoxon:.2e}" if not np.isnan(p_wilcoxon) else "nan"
            print(f"{pname:<18} {m:<10} {mean_0:.4f}     {mean_best:.4f}     {opt_alpha:<8.2f} {pct:+.1f}%      {s001:<8} {s005:<8} {s01:<8} {p_t_str:<12} {p_w_str:<12}")
            csv_rows.append([pname, m, f"{mean_0:.4f}", f"{mean_best:.4f}", f"{opt_alpha:.2f}", f"{pct:+.1f}%", s001 or "", s005 or "", s01, p_t_str, p_w_str])
        else:
            print(f"{pname:<18} {m:<10} {mean_0:.4f}     {mean_best:.4f}     {opt_alpha:<8.2f} {pct:+.1f}%      (no scipy)    (no scipy)    -")
            csv_rows.append([pname, m, f"{mean_0:.4f}", f"{mean_best:.4f}", f"{opt_alpha:.2f}", f"{pct:+.1f}%", "", "", "-", "-", "-"])

print("\n" + "="*120)
print("paired t-test / Wilcoxon, common seq_ids")
print("="*120)

out_path = REPO_ROOT / 'results' / 'folding' / 'alpha0_vs_best_full.csv'
with open(out_path, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['partition', 'model', 'mean(α=0)', 'mean(best)', 'best_α', '%Δ', 'p<0.001', 'p<0.005', 'p<0.01', 'p_ttest', 'p_wilcoxon'])
    w.writerows(csv_rows)
print(f"\nWrote {out_path}")
