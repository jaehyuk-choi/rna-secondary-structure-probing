#!/usr/bin/env python3
"""Bin CPLfold per-seq F1 by sequence length (default or custom bins)."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_seq_lengths(bpRNA_csv):
    out = {}
    with open(bpRNA_csv) as f:
        for r in csv.DictReader(f):
            seq = r.get('sequence', r.get('seq', ''))
            out[r['id']] = len(seq)
    return out


def get_bin(length, bins=None):
    if bins is None:
        if length < 100: return '<100'
        if length < 200: return '100-199'
        if length < 400: return '200-399'
        if length < 600: return '400-599'
        return '600+'
    for label, lo, hi in bins:
        if lo <= length < hi:
            return label
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default=str(REPO_ROOT / 'results' / 'folding'))
    ap.add_argument('--bpRNA-csv', default=str(REPO_ROOT / 'data' / 'metadata' / 'bpRNA.csv'))
    ap.add_argument('--output', default=None, help='Output CSV path')
    ap.add_argument('--bins', default='default',
                    help='default or custom: <100,200-400,400+ (format: label:lo:hi,label:lo:hi)')
    args = ap.parse_args()

    lengths = load_seq_lengths(args.bpRNA_csv)

    if args.bins == 'default':
        bin_specs = [('<100', 0, 100), ('100-199', 100, 200), ('200-399', 200, 400),
                     ('400-599', 400, 600), ('600+', 600, 99999)]
    elif args.bins == 'simple':
        bin_specs = [('<100', 0, 100), ('100-199', 100, 200), ('200-400', 200, 401), ('400+', 401, 99999)]
    else:
        bin_specs = []
        for part in args.bins.split(','):
            p = part.strip()
            if p == '<100':
                bin_specs.append(('<100', 0, 100))
            elif p == '200-400':
                bin_specs.append(('200-400', 200, 400))
            elif p == '400+':
                bin_specs.append(('400+', 400, 99999))
            else:
                toks = p.split(':')
                if len(toks) == 3:
                    bin_specs.append((toks[0], int(toks[1]), int(toks[2])))

    def bin_fn(L):
        for label, lo, hi in bin_specs:
            if lo <= L < hi:
                return label
        return 'other'

    results_dir = Path(args.results_dir)
    subdirs = ['ts0_vienna', 'ts0_contrafold', 'new_vienna', 'new_contrafold']
    models = ['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert']

    out_rows = []
    for subdir in subdirs:
        if not (results_dir / subdir).exists():
            continue
        backend = subdir.replace('_', ' ').title()
        for model in models:
            csv_path = results_dir / subdir / f'detailed_results_{model}.csv'
            if not csv_path.exists():
                continue
            by_bin = defaultdict(list)
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    sid = r['seq_id']
                    L = lengths.get(sid)
                    if L is None:
                        continue
                    b = bin_fn(L)
                    by_bin[b].append(float(r['f1']))
            order = {spec[0]: i for i, spec in enumerate(bin_specs)}
            for b in sorted(by_bin.keys(), key=lambda x: order.get(x, 99)):
                vals = by_bin[b]
                out_rows.append({
                    'partition_backend': subdir,
                    'model': model,
                    'length_bin': b,
                    'n': len(vals),
                    'mean_f1': sum(vals) / len(vals),
                })

    out_path = args.output or (results_dir.parent / 'results_by_length.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['partition_backend', 'model', 'length_bin', 'n', 'mean_f1'])
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
