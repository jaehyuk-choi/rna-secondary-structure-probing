#!/usr/bin/env python3
"""
Run CPLfold alpha sweep for TS0 and NEW partitions, with Vienna and Contrafold.

Calls feb8/scripts/run_split_pipeline.py for each:
  - TS0 + Vienna, TS0 + Contrafold
  - NEW + Vienna, NEW + Contrafold

For each model × backend combination.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-pairs-dir', default='/projects/u6cg/jay/dissertations/feb25/base_pairs_thresholded')
    ap.add_argument('--output-dir', default='/projects/u6cg/jay/dissertations/feb25/results_thresholded_ts0_new')
    ap.add_argument('--splits-csv', default='/projects/u6cg/jay/dissertations/data/bpRNA_splits.csv')
    ap.add_argument('--bpRNA-csv', default='/projects/u6cg/jay/dissertations/data/bpRNA.csv')
    ap.add_argument('--config-csv', default='/projects/u6cg/jay/dissertations/feb8/results_updated/summary/final_selected_config.csv')
    ap.add_argument('--alpha-start', type=float, default=0.0)
    ap.add_argument('--alpha-end', type=float, default=2.0)
    ap.add_argument('--alpha-step', type=float, default=0.02)
    ap.add_argument('--val-results-dir', default='/projects/u6cg/jay/dissertations/feb23/results_vl0_feb8',
                    help='VL0 Vienna dir for optimal alpha. If set, run only at optimal alpha (no sweep).')
    ap.add_argument('--val-contrafold-dir', default='/projects/u6cg/jay/dissertations/feb23/results_vl0_contrafold_feb8',
                    help='VL0 Contrafold dir for optimal alpha.')
    ap.add_argument('--num-workers', type=int, default=20)
    ap.add_argument('--models', nargs='+', default=['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot'])
    ap.add_argument('--partitions', nargs='+', default=['TS0', 'new'])
    ap.add_argument('--run-script', default='/projects/u6cg/jay/dissertations/feb8/scripts/run_split_pipeline.py')
    args = ap.parse_args()

    run_script = Path(args.run_script)
    if not run_script.exists():
        print(f"[ERROR] run_split_pipeline not found: {run_script}")
        return 1

    base_dir = Path(args.base_pairs_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # (partition, use_contrafold) -> output subdir
    configs = [
        ('TS0', False, 'ts0_vienna'),
        ('TS0', True, 'ts0_contrafold'),
        ('new', False, 'new_vienna'),
        ('new', True, 'new_contrafold'),
    ]

    for partition, use_contrafold, subdir in configs:
        out_dir = out_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        backend = 'Contrafold' if use_contrafold else 'Vienna'
        print(f"\n{'='*60}")
        print(f"Running: {partition} + {backend} -> {out_dir}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, str(run_script),
            '--partition', partition,
            '--base-pairs-dir', str(base_dir),
            '--output-dir', str(out_dir),
            '--splits-csv', args.splits_csv,
            '--bpRNA-csv', args.bpRNA_csv,
            '--config-csv', args.config_csv,
            '--alpha-start', str(args.alpha_start),
            '--alpha-end', str(args.alpha_end),
            '--alpha-step', str(args.alpha_step),
            '--num-workers', str(args.num_workers),
            '--models', *args.models,
        ]
        if use_contrafold:
            cmd.append('--use-contrafold')
            if args.val_contrafold_dir:
                cmd.extend(['--val-contrafold-dir', args.val_contrafold_dir])
        else:
            if args.val_results_dir:
                cmd.extend(['--val-results-dir', args.val_results_dir])

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[WARN] {partition} + {backend} returned {ret.returncode}")

    print(f"\n[INFO] All CPLfold runs complete. Output: {out_root}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
