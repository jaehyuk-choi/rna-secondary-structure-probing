#!/usr/bin/env python3
"""
Compare mean F1 at α=0 vs best α: % improvement and statistical significance.
For rnabert: α=0 baseline = same as other models (at α=0, bonus=0 → identical folding).
Uses paired t-test and Wilcoxon signed-rank.
"""
import csv
from pathlib import Path
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def load_seq_level(path, alpha, tol=0.01):
    """Load {seq_id: f1} for rows with alpha ~= alpha."""
    if not Path(path).exists(): return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            a = float(r['alpha'])
            if abs(a - alpha) < tol:
                out[r['seq_id']] = float(r['f1'])
    return out

def load_seq_level_any(path):
    """Load {seq_id: f1} - feb25 has single alpha per model."""
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

FEB25 = Path('/projects/u6cg/jay/dissertations/feb25/results_thresholded_ts0_new')
FEB8 = {
    'ts0_vienna': Path('/projects/u6cg/jay/dissertations/feb8/results_ts0'),
    'ts0_contrafold': Path('/projects/u6cg/jay/dissertations/feb8/results_ts0_contrafold'),
    'new_vienna': Path('/projects/u6cg/jay/dissertations/feb8/results_new'),
    'new_contrafold': Path('/projects/u6cg/jay/dissertations/feb8/results_new_contrafold'),
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
print("α=0 vs best α: % improvement, mean F1, and statistical significance")
print("rnabert α=0 baseline = same as other models (bonus=0 → identical structure)")
print("="*90)

for pkey, pname in partitions:
    print(f"\n### {pname} ###")
    backend = 'Vienna' if 'vienna' in pkey else 'Contrafold'
    feb8_dir = FEB8.get(pkey)

    # α=0 baseline for rnabert: use same as other models (partition mean at α=0)
    f1_0_ref = load_seq_level(feb8_dir / f'detailed_results_{ALPHA0_REF_MODEL}.csv', 0.0) if feb8_dir else {}
    mean_alpha0_partition = np.mean(list(f1_0_ref.values())) if f1_0_ref else float('nan')

    print(f"{'partition':<18} {'model':<10} {'mean(α=0)':<10} {'mean(best)':<10} {'best_α':<8} {'%Δ':<10} {'p<0.001':<8} {'p<0.005':<8} {'p<0.01':<8} {'p_ttest':<12} {'p_wilcoxon':<12}")
    print("-"*120)

    for m in models:
        opt_alpha = VAL_OPT.get((m, backend))
        d25 = load_seq_level_any(FEB25 / pkey / f'detailed_results_{m}.csv')
        if m == 'rnabert':
            d0 = f1_0_ref  # seq-level from ernie (same folding at α=0)
        else:
            d0 = load_seq_level(feb8_dir / f'detailed_results_{m}.csv', 0.0) if feb8_dir else {}

        common = sorted(set(d0.keys()) & set(d25.keys()))
        if not common:
            print(f"{pname:<18} {m:<10} -          -          -        -          -            -            -")
            continue

        arr_0 = np.array([d0[s] for s in common])
        arr_best = np.array([d25[s] for s in common])
        mean_0 = mean_alpha0_partition if m == 'rnabert' else arr_0.mean()  # rnabert: use partition α=0
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
print("METHODOLOGY:")
print("  - Paired design: same seq_ids in both α=0 (feb8) and best α (feb25)")
print("  - Paired t-test: H0: mean(arr_best - arr_0) = 0, two-tailed")
print("  - Wilcoxon signed-rank: non-parametric paired test (same H0)")
print("  - Significance: *** p<0.001, ** p<0.005, * p<0.01, n.s. not significant")
print("  - p=nan: when arr_0 ≈ arr_best (e.g. rinalmo best_α=0 → no difference)")
print("="*120)

# Write CSV
out_path = Path(__file__).parent.parent / 'alpha0_vs_best_full.csv'
with open(out_path, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['partition', 'model', 'mean(α=0)', 'mean(best)', 'best_α', '%Δ', 'p<0.001', 'p<0.005', 'p<0.01', 'p_ttest', 'p_wilcoxon'])
    w.writerows(csv_rows)
print(f"\n[INFO] Wrote {out_path}")
