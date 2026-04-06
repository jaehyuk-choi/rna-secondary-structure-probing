#!/usr/bin/env python3
"""CSV table: model canonical rates vs sequence-composition baseline (WC / WC+GU)."""

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_LABELS = {'ernie': 'ERNIE', 'roberta': 'RoBERTa', 'rnafm': 'RNAFM',
                'rinalmo': 'RiNALMo', 'onehot': 'One-hot', 'rnabert': 'RNABERT'}


def main():
    baseline_csv = REPO_ROOT / 'results' / 'tables' / 'canonical_rate_baseline_table.csv'
    if not baseline_csv.exists():
        subprocess.run([sys.executable,
                        str(REPO_ROOT / 'code' / 'analysis' / 'compute_wc_gu_rate_all_pairs.py')],
                       check=True, cwd=str(REPO_ROOT))

    ts0 = {}
    with open(REPO_ROOT / 'results' / 'metrics' / 'final_test_metrics_wobble.csv') as f:
        for r in csv.DictReader(f):
            ts0[r['model']] = {'wc': float(r['canonical_rate']), 'wc_gu': float(r['canonical_rate_wobble'])}
    new = {}
    with open(REPO_ROOT / 'results' / 'metrics' / 'final_new_metrics_wobble.csv') as f:
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

    out_csv = REPO_ROOT / 'results' / 'tables' / 'canonical_rate_table_with_baseline.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'ts0_wc', 'ts0_wc_gu', 'new_wc', 'new_wc_gu'])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {out_csv}")
    print("\n--- Table ---")
    for r in rows:
        print(f"  {r['model']:25} | TS0: {r['ts0_wc']:>6} {r['ts0_wc_gu']:>6} | NEW: {r['new_wc']:>6} {r['new_wc_gu']:>6}")


if __name__ == '__main__':
    main()
