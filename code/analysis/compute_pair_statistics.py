#!/usr/bin/env python3
"""Per-split pair counts, pair rate, AU/GC/GU mix → results/tables/pair_statistics_by_split.csv."""

import ast
import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SPLITS_CSV = REPO_ROOT / 'data' / 'splits' / 'bpRNA_splits.csv'
BPRNA_CSV = REPO_ROOT / 'data' / 'metadata' / 'bpRNA.csv'
OUT_DIR = REPO_ROOT / 'results' / 'tables'

CANONICAL_WC = {'AU', 'UA', 'GC', 'CG'}
WOBBLE = {'GU', 'UG'}
PARTITIONS = ['TR0', 'VL0', 'TS0', 'NEW']


def main():
    # Load partition assignments
    partition_of = {}
    with open(SPLITS_CSV) as f:
        for row in csv.DictReader(f):
            partition_of[row['id']] = row['partition'].strip().upper()

    # Accumulate per-partition statistics
    stats = {p: {'n_seq': 0, 'n_pairs': 0, 'total_nt': 0, 'pair_types': Counter()}
             for p in PARTITIONS}

    with open(BPRNA_CSV) as f:
        for row in csv.DictReader(f):
            sid = row['id']
            part = partition_of.get(sid, '')
            if part not in stats:
                continue
            seq = row['sequence']
            bp_list = ast.literal_eval(row['base_pairs'])
            s = stats[part]
            s['n_seq'] += 1
            s['total_nt'] += len(seq)
            s['n_pairs'] += len(bp_list)
            for i, j in bp_list:
                pair = seq[i] + seq[j]
                s['pair_types'][pair.upper()] += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'pair_statistics_by_split.csv'
    fieldnames = ['partition', 'n_sequences', 'total_pairs', 'mean_length',
                  'pair_rate', 'AU_frac', 'GC_frac', 'GU_frac',
                  'canonical_wc_frac', 'canonical_wc_gu_frac']

    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for p in PARTITIONS:
            s = stats[p]
            n_pairs = s['n_pairs']
            n_seq = s['n_seq']
            mean_len = s['total_nt'] / n_seq if n_seq else 0
            # Pair rate: pairs per possible pair (L*(L-1)/2)
            total_possible = sum(
                (l * (l - 1)) / 2
                for l in [len(row['sequence'])
                          for row in []]  # placeholder
            ) if False else 0
            # Simpler: pairs per nucleotide
            pair_rate = n_pairs / s['total_nt'] if s['total_nt'] else 0

            ct = s['pair_types']
            au = ct.get('AU', 0) + ct.get('UA', 0)
            gc = ct.get('GC', 0) + ct.get('CG', 0)
            gu = ct.get('GU', 0) + ct.get('UG', 0)
            wc = au + gc
            wc_gu = wc + gu

            w.writerow({
                'partition': p,
                'n_sequences': n_seq,
                'total_pairs': n_pairs,
                'mean_length': f'{mean_len:.1f}',
                'pair_rate': f'{pair_rate:.4f}',
                'AU_frac': f'{au / n_pairs:.4f}' if n_pairs else '0',
                'GC_frac': f'{gc / n_pairs:.4f}' if n_pairs else '0',
                'GU_frac': f'{gu / n_pairs:.4f}' if n_pairs else '0',
                'canonical_wc_frac': f'{wc / n_pairs:.4f}' if n_pairs else '0',
                'canonical_wc_gu_frac': f'{wc_gu / n_pairs:.4f}' if n_pairs else '0',
            })

    print(f"Wrote {out_path}")
    # Print summary
    for p in PARTITIONS:
        s = stats[p]
        print(f"  {p}: {s['n_seq']} seqs, {s['n_pairs']} pairs")


if __name__ == '__main__':
    main()
