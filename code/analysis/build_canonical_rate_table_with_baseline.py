#!/usr/bin/env python3
"""
표 형식: Canonical rate (WC, WC+GU) by model + Baseline.
Baseline = 모든 후보쌍 (i<j) 중 WC/WC+GU 비율.
"""

import csv
import subprocess
import sys
from pathlib import Path

MARCH1 = Path(__file__).resolve().parents[1]
MODEL_LABELS = {'ernie': 'ERNIE', 'roberta': 'RoBERTa', 'rnafm': 'RNAFM',
                'rinalmo': 'RiNALMo', 'onehot': 'One-hot', 'rnabert': 'RNABERT'}


def main():
    # Run baseline computation if needed
    baseline_csv = MARCH1 / 'data' / 'canonical_rate_baseline_table.csv'
    if not baseline_csv.exists():
        subprocess.run([sys.executable, str(MARCH1 / 'scripts' / 'compute_wc_gu_rate_all_pairs.py')],
                      check=True, cwd=str(MARCH1))

    ts0 = {}
    with open(MARCH1 / 'final_test_metrics_wobble.csv') as f:
        for r in csv.DictReader(f):
            ts0[r['model']] = {'wc': float(r['canonical_rate']), 'wc_gu': float(r['canonical_rate_wobble'])}
    new = {}
    with open(MARCH1 / 'final_new_metrics_wobble.csv') as f:
        for r in csv.DictReader(f):
            new[r['model']] = {'wc': float(r['canonical_rate']), 'wc_gu': float(r['canonical_rate_wobble'])}

    with open(baseline_csv) as f:
        bl = next(csv.DictReader(f))

    rows = []
    for m in ['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert']:
        t = ts0.get(m, {})
        n = new.get(m, {})
        rows.append({
            'model': MODEL_LABELS.get(m, m),
            'ts0_wc': f"{100*t.get('wc',0):.1f}%",
            'ts0_wc_gu': f"{100*t.get('wc_gu',0):.1f}%",
            'new_wc': f"{100*n.get('wc',0):.1f}%",
            'new_wc_gu': f"{100*n.get('wc_gu',0):.1f}%",
        })
    rows.append({
        'model': 'Baseline (all pairs)',
        'ts0_wc': bl['ts0_wc_pct'],
        'ts0_wc_gu': bl['ts0_wc_gu_pct'],
        'new_wc': bl['new_wc_pct'],
        'new_wc_gu': bl['new_wc_gu_pct'],
    })

    out_csv = MARCH1 / 'data' / 'canonical_rate_table_with_baseline.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'ts0_wc', 'ts0_wc_gu', 'new_wc', 'new_wc_gu'])
        w.writeheader()
        w.writerows(rows)

    # LaTeX table
    out_tex = MARCH1 / 'data' / 'canonical_rate_table_with_baseline.tex'
    with open(out_tex, 'w') as f:
        f.write(r"""\begin{table*}[t]
\centering
\caption{Canonical base-pair rate (\% of predicted pairs that are Watson--Crick or GU wobble) under unconstrained decoding. Baseline: proportion of WC/WC+GU among all candidate pairs $(i<j)$ in the data.}
\label{tab:canonical_ts_new}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{TS0} & \multicolumn{2}{c}{NEW} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Model & WC & WC+GU & WC & WC+GU \\
\midrule
""")
        for r in rows:
            # LaTeX: % -> \%
            def tex_pct(s): return s.replace('%', r'\%')
            f.write(f"{r['model']} & {tex_pct(r['ts0_wc'])} & {tex_pct(r['ts0_wc_gu'])} & {tex_pct(r['new_wc'])} & {tex_pct(r['new_wc_gu'])} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table*}
""")

    print(f"Saved {out_csv}")
    print(f"Saved {out_tex}")
    print("\n--- Table ---")
    for r in rows:
        print(f"  {r['model']:25} | TS0: {r['ts0_wc']:>6} {r['ts0_wc_gu']:>6} | NEW: {r['new_wc']:>6} {r['new_wc_gu']:>6}")


if __name__ == '__main__':
    main()
