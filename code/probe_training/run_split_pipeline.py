#!/usr/bin/env python3
"""Run CPLfold alpha sweep for a partition (TS0, VL0, new).

This is a feb23-local copy of feb8/scripts/run_split_pipeline.py with one critical fix:
- Robustly parse CPLfold_inter output lines of the form: "<dotbracket> (<energy>)".
  The previous split("(")[0] logic truncates structures (dot-bracket contains '('),
  causing almost all rows to be dropped and CSVs to stay empty.

Outputs: detailed_results_{model}.csv in output-dir.
"""

import argparse
import ast
import csv
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

sys.path.insert(0, '/projects/u6cg/jay/dissertations/jan22')
from utils.evaluation import compute_pair_metrics, precision_recall_f1


def _get_python_for_subprocess():
    return sys.executable


def dot_bracket_to_pairs(structure, seq_length):
    pairs = []
    stack = []
    for i, char in enumerate(structure):
        pos = i + 1
        if char == '(':
            stack.append(pos)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((min(j, pos), max(j, pos)))
    return sorted(pairs)


def parse_base_pairs(base_pairs_str):
    try:
        pairs_list = ast.literal_eval(base_pairs_str)
        return sorted(
            [(min(int(p[0]), int(p[1])), max(int(p[0]), int(p[1]))) for p in pairs_list if len(p) >= 2]
        )
    except Exception:
        return []


def pairs_to_contact_map(pairs, seq_length):
    m = torch.zeros(seq_length, seq_length, dtype=torch.float32)
    for i, j in pairs:
        m[i - 1, j - 1] = 1.0
        m[j - 1, i - 1] = 1.0
    return m


def load_partition_seqs(splits_csv, partition):
    ids = []
    with open(splits_csv) as f:
        for row in csv.DictReader(f):
            p = row.get('partition', '').strip()
            if p.upper() == partition.upper():
                ids.append(row['id'])
    return ids


def load_bpRNA(bpRNA_csv):
    data = {}
    with open(bpRNA_csv) as f:
        for row in csv.DictReader(f):
            data[row['id']] = {
                'sequence': row.get('sequence', row.get('seq', '')),
                'base_pairs': row.get('base_pairs', '[]'),
            }
    return data


def load_config(config_csv, models):
    cfg = {}
    with open(config_csv) as f:
        for row in csv.DictReader(f):
            m = row['model']
            if m in models:
                cfg[m] = {
                    'layer': int(row['selected_layer']),
                    'k': int(row['selected_k']),
                    'threshold': float(row['selected_best_threshold']),
                }
    return cfg


def _alpha_grid(alpha_start: float, alpha_end: float, alpha_step: float):
    if alpha_step <= 0:
        raise ValueError('alpha_step must be > 0')
    n = int(round((alpha_end - alpha_start) / alpha_step))
    # include endpoint if it lands on the grid within rounding
    alphas = [round(alpha_start + i * alpha_step, 2) for i in range(n + 1) if alpha_start + i * alpha_step <= alpha_end + 1e-9]
    return alphas


def _parse_cplfold_struct_energy(stdout: str):
    """Return (structure, energy) from CPLfold_inter stdout.

    CPLfold_inter prints:
      line0: seq
      line1: "<dotbracket> (<energy>)"

    Some modes may print extra lines; we search from the end for a line containing " (" and ending with ")".
    """
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for line in reversed(lines):
        if ' (' not in line or not line.endswith(')'):
            continue
        struct_part, energy_part = line.rsplit(' (', 1)
        if not struct_part:
            continue
        try:
            energy = float(energy_part[:-1].strip())
        except Exception:
            continue
        return struct_part.strip(), energy
    return None, None


def run_cplfold(seq, bonus_file, alpha, use_contrafold=False, cplfold_path=None, python_exe=None, timeout_s=180):
    cplfold_path = cplfold_path or '/projects/u6cg/jay/CPLfold_inter/CPLfold_inter.py'
    python_exe = python_exe or _get_python_for_subprocess()
    cmd = [
        python_exe, cplfold_path, seq,
        '--bonus', str(bonus_file),
        '--alpha', str(alpha),
    ]
    if not use_contrafold:
        cmd.append('--V')

    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        struct, energy = _parse_cplfold_struct_energy(out.stdout)
        if struct is not None:
            return struct, energy

        if out.returncode != 0 and out.stderr:
            print(f"[WARN] CPLfold failed rc={out.returncode}: {out.stderr[:300]}", file=sys.stderr, flush=True)
        return None, None
    except subprocess.TimeoutExpired:
        print(f"[WARN] CPLfold timeout after {timeout_s}s (len={len(seq)})", file=sys.stderr, flush=True)
        return None, None
    except Exception as e:
        print(f"[WARN] CPLfold exception: {e}", file=sys.stderr, flush=True)
        return None, None


def process_one(args):
    seq_id, sequence, true_pairs, model, alpha, bonus_file, use_contrafold, threshold, timeout_s = args
    seq_len = len(sequence)
    true_contact = pairs_to_contact_map(true_pairs, seq_len)

    struct, energy = run_cplfold(sequence, bonus_file, alpha, use_contrafold, timeout_s=timeout_s)
    if (not struct) or (len(struct) != seq_len):
        return None

    pred_pairs = dot_bracket_to_pairs(struct, seq_len)
    pred_contact = pairs_to_contact_map(pred_pairs, seq_len)

    TP, FP, FN = compute_pair_metrics(
        pred_contact, true_contact,
        threshold=0.0, shift=1, inputs_are_logits=False
    )
    prec, rec, f1 = precision_recall_f1(TP, FP, FN)

    return {
        'seq_id': seq_id,
        'model': model,
        'alpha': alpha,
        'threshold_used': threshold,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'tp': TP,
        'fp': FP,
        'fn': FN,
        'predicted_count': len(pred_pairs),
        'energy': energy,
        'structure': struct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partition', required=True, help='VL0, TS0, or new')
    ap.add_argument('--base-pairs-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--splits-csv', default='/projects/u6cg/jay/dissertations/data/bpRNA_splits.csv')
    ap.add_argument('--bpRNA-csv', default='/projects/u6cg/jay/dissertations/data/bpRNA.csv')
    ap.add_argument('--config-csv', default='/projects/u6cg/jay/dissertations/jan22/results_updated/summary/final_selected_config.csv')
    ap.add_argument('--alpha-start', type=float, default=0.0)
    ap.add_argument('--alpha-end', type=float, default=2.0)
    ap.add_argument('--alpha-step', type=float, default=0.02)
    ap.add_argument('--num-workers', type=int, default=20)
    ap.add_argument('--timeout-s', type=int, default=180)
    ap.add_argument('--use-contrafold', action='store_true')
    ap.add_argument('--models', nargs='+', default=['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert'])
    args = ap.parse_args()

    base_dir = Path(args.base_pairs_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_ids = load_partition_seqs(args.splits_csv, args.partition)
    print(f"[INFO] Loading {args.partition} sequences from: {args.splits_csv}")
    print(f"[INFO] Found {len(seq_ids)} {args.partition} sequences")

    bpRNA = load_bpRNA(args.bpRNA_csv)
    print(f"[INFO] Loaded {len(bpRNA)} sequences from bpRNA.csv")

    config = load_config(args.config_csv, args.models)
    print(f"[INFO] Loading optimal configurations from: {args.config_csv}")

    alphas = _alpha_grid(args.alpha_start, args.alpha_end, args.alpha_step)
    print(f"[INFO] Alpha values: {len(alphas)} points from {alphas[0]:.2f} to {alphas[-1]:.2f} (step={args.alpha_step})")

    for model in args.models:
        if model not in config:
            continue
        cfg = config[model]
        thresh = cfg['threshold']
        bonus_dir = base_dir / model

        csv_path = out_dir / f'detailed_results_{model}.csv'
        existing = set()
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with open(csv_path) as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        key = (row['seq_id'], float(row['alpha']))
                    except Exception:
                        continue
                    existing.add(key)
            print(f"[INFO] Found {len(existing)} existing rows in {csv_path.name}")

        tasks = []
        for seq_id in seq_ids:
            if seq_id not in bpRNA:
                continue
            seq = bpRNA[seq_id]['sequence']
            if not seq:
                continue
            try:
                pairs_raw = bpRNA[seq_id]['base_pairs']
                true_pairs = parse_base_pairs(pairs_raw) if isinstance(pairs_raw, str) else pairs_raw
            except Exception:
                true_pairs = []

            bonus_file = bonus_dir / f'base_pair_{model}_{seq_id}.txt'
            if not bonus_file.exists():
                continue

            for alpha in alphas:
                if (seq_id, alpha) in existing:
                    continue
                tasks.append((seq_id, seq, true_pairs, model, alpha, str(bonus_file), args.use_contrafold, thresh, args.timeout_s))

        if not tasks:
            print(f"[INFO] {model}: no new tasks")
            continue

        print(f"[INFO] Processing {model}: {len(tasks)} tasks with {args.num_workers} workers")
        fieldnames = ['seq_id', 'model', 'alpha', 'threshold_used', 'f1', 'precision', 'recall', 'tp', 'fp', 'fn', 'predicted_count', 'energy', 'structure']
        file_has_data = csv_path.exists() and csv_path.stat().st_size > 0

        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_has_data:
                w.writeheader()
                f.flush()

            with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                futures = {ex.submit(process_one, t): t for t in tasks}
                done = 0
                wrote = 0
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        w.writerow(res)
                        f.flush()
                        wrote += 1
                    done += 1
                    if done % 500 == 0:
                        print(f"[INFO] {model}: {done}/{len(tasks)} done (rows_written={wrote})")

        print(f"[INFO] {model}: done. Output: {csv_path}")

    print("\nAll done!")


if __name__ == '__main__':
    raise SystemExit(main())
