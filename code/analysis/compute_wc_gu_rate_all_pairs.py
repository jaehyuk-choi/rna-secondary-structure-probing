#!/usr/bin/env python3
"""
데이터에서 "모든 후보쌍 (i<j)" 중 WC+GU가 차지하는 비율 계산.

각 시퀀스에 대해: 모든 (i,j) where i<j 에서
- base_i, base_j가 WC(AU,UA,CG,GC) 또는 GU(GU,UG)이면 canonical
- rate = canonical_count / total_pairs
"""

import csv
from pathlib import Path

BASES = ['A', 'U', 'G', 'C']
WC_ONLY = {('A','U'), ('U','A'), ('G','C'), ('C','G')}  # Watson-Crick only
WC_GU = WC_ONLY | {('G','U'), ('U','G')}  # WC + wobble


def count_pairs(seq: str, pair_set: set) -> int:
    """Count (i,j) with i<j where (seq[i], seq[j]) in pair_set. O(L) via suffix counts."""
    from collections import Counter
    partner = {}
    for b1, b2 in pair_set:
        for b in (b1.lower(), b1.upper()):
            partner.setdefault(b, []).append(b2)
    suffix = Counter(seq)
    count = 0
    for i in range(len(seq)):
        b = seq[i]
        for p in partner.get(b, []):
            count += suffix.get(p, 0)
        suffix[b] = suffix.get(b, 0) - 1
        if suffix[b] <= 0:
            del suffix[b]
    return count


def main():
    bpRNA = {}
    with open('/projects/u6cg/jay/dissertations/data/bpRNA.csv') as f:
        for row in csv.DictReader(f):
            bpRNA[row['id']] = row['sequence']

    seq_ids = []
    with open('/projects/u6cg/jay/dissertations/data/bpRNA_splits.csv') as f:
        for row in csv.DictReader(f):
            p = row.get('partition', '').strip().upper()
            if p in ('TS0', 'NEW'):
                seq_ids.append(row['id'])

    by_partition = {'TS0': (0, 0, 0), 'NEW': (0, 0, 0)}  # total_pairs, wc_count, wc_gu_count
    partition_map = {}
    with open('/projects/u6cg/jay/dissertations/data/bpRNA_splits.csv') as f:
        for row in csv.DictReader(f):
            partition_map[row['id']] = row.get('partition', '').strip().upper()

    for seq_id in seq_ids:
        if seq_id not in bpRNA:
            continue
        seq = bpRNA[seq_id]
        L = len(seq)
        n_pairs = L * (L - 1) // 2
        wc_count = count_pairs(seq, WC_ONLY)
        wc_gu_count = count_pairs(seq, WC_GU)
        part = partition_map.get(seq_id, '')
        if part in by_partition:
            p, wc, wcg = by_partition[part]
            by_partition[part] = (p + n_pairs, wc + wc_count, wcg + wc_gu_count)

    out_dir = Path('/projects/u6cg/jay/dissertations/march1/data')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Table format: Model, TS0_WC, TS0_WC+GU, NEW_WC, NEW_WC+GU (as %)
    ts0_pairs, ts0_wc, ts0_wc_gu = by_partition['TS0']
    new_pairs, new_wc, new_wc_gu = by_partition['NEW']
    ts0_wc_gu_rate = 100*ts0_wc_gu/ts0_pairs if ts0_pairs else 0
    new_wc_gu_rate = 100*new_wc_gu/new_pairs if new_pairs else 0
    baseline_row = {
        'model': 'Baseline (all pairs)',
        'ts0_wc_pct': f'{100*ts0_wc/ts0_pairs:.1f}%' if ts0_pairs else '-',
        'ts0_wc_gu_pct': f'{ts0_wc_gu_rate:.2f}%' if ts0_pairs else '-',
        'new_wc_pct': f'{100*new_wc/new_pairs:.1f}%' if new_pairs else '-',
        'new_wc_gu_pct': f'{new_wc_gu_rate:.2f}%' if new_pairs else '-',
    }

    # Save baseline table (LaTeX-ready format matching the canonical rate table)
    table_path = out_dir / 'canonical_rate_baseline_table.csv'
    with open(table_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'ts0_wc_pct', 'ts0_wc_gu_pct', 'new_wc_pct', 'new_wc_gu_pct'])
        w.writeheader()
        w.writerow(baseline_row)

    # Also save raw rates
    raw_path = out_dir / 'wc_gu_rate_all_pairs.csv'
    rows = [
        {'partition': 'TS0', 'total_pairs': ts0_pairs, 'wc_count': ts0_wc, 'wc_gu_count': ts0_wc_gu,
         'wc_rate': ts0_wc/ts0_pairs if ts0_pairs else 0, 'wc_gu_rate': ts0_wc_gu/ts0_pairs if ts0_pairs else 0},
        {'partition': 'NEW', 'total_pairs': new_pairs, 'wc_count': new_wc, 'wc_gu_count': new_wc_gu,
         'wc_rate': new_wc/new_pairs if new_pairs else 0, 'wc_gu_rate': new_wc_gu/new_pairs if new_pairs else 0},
    ]
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['partition', 'total_pairs', 'wc_count', 'wc_gu_count', 'wc_rate', 'wc_gu_rate'])
        w.writeheader()
        w.writerows(rows)

    print("--- Baseline: 모든 후보쌍 (i<j) 중 WC / WC+GU 비율 ---")
    print(f"  TS0: WC {100*ts0_wc/ts0_pairs:.1f}%, WC+GU {ts0_wc_gu:,} / {ts0_pairs:,} = {ts0_wc_gu_rate:.2f}%")
    print(f"  NEW: WC {100*new_wc/new_pairs:.1f}%, WC+GU {new_wc_gu:,} / {new_pairs:,} = {new_wc_gu_rate:.2f}%")
    print(f"\n표 형식 (Baseline 행):")
    print(f"  {baseline_row['model']} | TS0 WC {baseline_row['ts0_wc_pct']} | TS0 WC+GU {baseline_row['ts0_wc_gu_pct']} | NEW WC {baseline_row['new_wc_pct']} | NEW WC+GU {baseline_row['new_wc_gu_pct']}")
    print(f"\nSaved {table_path}, {raw_path}")


if __name__ == '__main__':
    main()
