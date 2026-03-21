#!/usr/bin/env python3
"""
Build unconstrained_results_summary.csv from final_test_metrics.csv and final_new_metrics.csv.
Run after probe_only completes.
"""

import csv
from pathlib import Path

MARCH1 = Path('/projects/u6cg/jay/dissertations/march1')
CONFIG = MARCH1 / 'final_selected_config_unconstrained.csv'


def main():
    config = {}
    with open(CONFIG) as f:
        for row in csv.DictReader(f):
            config[row['model']] = {
                'layer': row['selected_layer'],
                'k': row['selected_k'],
                'threshold': row['selected_best_threshold'],
                'decoding_mode': row['selected_decoding_mode'],
            }

    # We don't have val_f1 in config - need to get from val files or leave blank
    ts0 = {}
    with open(MARCH1 / 'final_test_metrics.csv') as f:
        for row in csv.DictReader(f):
            ts0[row['model']] = row
    new = {}
    with open(MARCH1 / 'final_new_metrics.csv') as f:
        for row in csv.DictReader(f):
            new[row['model']] = row

    summary_path = MARCH1 / 'unconstrained_results_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        fieldnames = [
            'model', 'layer', 'k', 'threshold', 'decoding_mode',
            'ts0_f1', 'ts0_precision', 'ts0_recall',
            'new_f1', 'new_precision', 'new_recall',
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in sorted(config.keys()):
            cfg = config[m]
            t = ts0.get(m, {})
            n = new.get(m, {})
            w.writerow({
                'model': m,
                'layer': cfg['layer'],
                'k': cfg['k'],
                'threshold': cfg['threshold'],
                'decoding_mode': cfg['decoding_mode'],
                'ts0_f1': t.get('f1', ''),
                'ts0_precision': t.get('precision', ''),
                'ts0_recall': t.get('recall', ''),
                'new_f1': n.get('f1', ''),
                'new_precision': n.get('precision', ''),
                'new_recall': n.get('recall', ''),
            })
    print(f"Wrote {summary_path}")


if __name__ == '__main__':
    main()
