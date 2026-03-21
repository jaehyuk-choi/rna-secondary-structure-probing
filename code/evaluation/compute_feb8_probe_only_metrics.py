#!/usr/bin/env python3
"""
Compute TS0 and NEW probe-only (threshold sweep) metrics for feb8 selected configs.
Updates final_test_metrics.csv and final_new_metrics.csv with feb8 config values.

Uses CPU only (device='cpu' throughout; no CUDA/GPU).
"""

import argparse
import ast
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/projects/u6cg/jay/dissertations/jan22')
from utils.evaluation import (
    compute_canonical_rate,
    compute_pair_metrics,
    contact_to_pairs,
    create_canonical_mask,
    precision_recall_f1,
    prob_to_pairs,
)

sys.path.insert(0, '/projects/u6cg/jay/dissertations/jan22')
from scripts.generation.generate_base_pairs import load_probe_matrix


def get_embedding_path(seq_id, model, layer, embeddings_base):
    emb_base = Path(embeddings_base)
    model_dirs = {
        'ernie': 'RNAErnie', 'roberta': 'RoBERTa', 'rnafm': 'RNAFM',
        'rinalmo': 'RiNALMo', 'onehot': 'Onehot', 'rnabert': 'RNABERT',
    }
    model_dir = model_dirs.get(model, model)
    return emb_base / model_dir / 'bpRNA' / 'by_layer' / f'layer_{layer}' / f'{seq_id}.npy'


def parse_base_pairs(base_pairs_str):
    try:
        pairs_list = ast.literal_eval(base_pairs_str)
        return sorted([(min(int(p[0]), int(p[1])), max(int(p[0]), int(p[1]))) for p in pairs_list if len(p) >= 2])
    except Exception:
        return []


def pairs_to_contact_map(pairs, seq_length):
    m = torch.zeros(seq_length, seq_length, dtype=torch.float32)
    for i, j in pairs:
        m[i - 1, j - 1] = 1.0
        m[j - 1, i - 1] = 1.0
    return m


def get_allowed_mask(decoding_mode, sequence, device='cpu'):
    if decoding_mode == 'unconstrained':
        return None
    allow_gu = (decoding_mode == 'canonical_wobble')
    return create_canonical_mask(sequence, allow_gu=allow_gu, device=device)


def eval_one_seq(seq_id, sequence, true_pairs, emb_path, B, thresh, decoding_mode, model=None, device='cpu'):
    """Compute F1 for one sequence with given config."""
    if not Path(emb_path).exists():
        return None
    try:
        emb = np.load(emb_path)
        emb_t = torch.from_numpy(emb).float().to(device)
        z = torch.matmul(emb_t, B.t())
        p_ij = torch.sigmoid(torch.matmul(z, z.t()))
        # RNAFM adds BOS/EOS at END: emb has seq_len+2 positions; bases are first L, use [0:-2, 0:-2]
        if model == 'rnafm' and p_ij.shape[0] == len(sequence) + 2:
            p_ij = p_ij[0:-2, 0:-2]
        L = p_ij.shape[0]
        allowed_mask = get_allowed_mask(decoding_mode, sequence, device)
        pred_pairs = prob_to_pairs(p_ij, threshold=thresh, allowed_mask=allowed_mask, inputs_are_logits=False)
        true_contact = pairs_to_contact_map(true_pairs, L)
        pred_contact = pairs_to_contact_map(pred_pairs, L)
        TP, FP, FN = compute_pair_metrics(pred_contact, true_contact, threshold=0.0, shift=1, inputs_are_logits=False)
        prec, rec, f1 = precision_recall_f1(TP, FP, FN)
        canonical_rate = float('nan')
        if pred_pairs and sequence:
            canonical_rate = compute_canonical_rate(pred_pairs, sequence, allow_gu=(decoding_mode == 'canonical_wobble'))
        return {'TP': TP, 'FP': FP, 'FN': FN, 'precision': prec, 'recall': rec, 'f1': f1, 'canonical_rate': canonical_rate}
    except Exception as e:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-csv', default='/projects/u6cg/jay/dissertations/feb8/results_updated/summary/final_selected_config.csv')
    ap.add_argument('--splits-csv', default='/projects/u6cg/jay/dissertations/data/bpRNA_splits.csv')
    ap.add_argument('--bpRNA-csv', default='/projects/u6cg/jay/dissertations/data/bpRNA.csv')
    ap.add_argument('--embeddings-base', default='/projects/u6cg/jay/dissertations/data/embeddings')
    ap.add_argument('--checkpoint-base', default='/projects/u6cg/jay/dissertations/feb8/results_updated/outputs')
    ap.add_argument('--output-dir', default='/projects/u6cg/jay/dissertations/feb8/results_updated/summary')
    ap.add_argument('--progress-log', default='', help='Path to progress log file (e.g. tail -f to monitor)')
    ap.add_argument('--models', nargs='+', default=['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert'])
    ap.add_argument('--overwrite', action='store_true', help='Load existing CSVs and overwrite only specified models (e.g. --models rnafm --overwrite)')
    ap.add_argument('--dataset', choices=['ts0', 'new', 'both'], default='both',
                    help='ts0: only TS0; new: only NEW (loads TS0 from existing CSV); both: both (default)')
    ap.add_argument('--resume', action='store_true',
                    help='Load existing CSVs, skip (model,partition) already done, compute only missing. Use with --models to limit scope.')
    args = ap.parse_args()

    # Load config
    config = {}
    with open(args.config_csv) as f:
        for row in csv.DictReader(f):
            m = row['model']
            if m in args.models:
                config[m] = {
                    'layer': int(row['selected_layer']),
                    'k': int(row['selected_k']),
                    'threshold': float(row['selected_best_threshold']),
                    'decoding_mode': row['selected_decoding_mode'],
                }

    # Load bpRNA
    bpRNA = {}
    with open(args.bpRNA_csv) as f:
        for row in csv.DictReader(f):
            bpRNA[row['id']] = {'sequence': row['sequence'], 'base_pairs': row['base_pairs']}

    # Load partition IDs
    ts0_ids = []
    new_ids = []
    with open(args.splits_csv) as f:
        for row in csv.DictReader(f):
            p = row.get('partition', '').strip().upper()
            if p == 'TS0':
                ts0_ids.append(row['id'])
            elif p == 'NEW':
                new_ids.append(row['id'])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_log = Path(args.progress_log) if args.progress_log else out_dir / 'probe_only_progress.log'

    def log(msg):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if progress_log:
            with open(progress_log, 'a') as f:
                f.write(line + '\n')
                f.flush()

    log(f"TS0: {len(ts0_ids)} sequences, NEW: {len(new_ids)} sequences")
    log(f"Progress log: {progress_log} (tail -f to monitor)")

    fieldnames = ['model', 'layer', 'k', 'seed', 'threshold', 'decoding_mode', 'precision', 'recall', 'f1', 'TP', 'FP', 'FN', 'canonical_rate']

    # Load existing results when --overwrite, --dataset new/ts0, or --resume
    results_ts0 = []
    results_new = []
    test_path = out_dir / 'final_test_metrics.csv'
    new_path = out_dir / 'final_new_metrics.csv'

    if args.resume:
        # Load all existing; we will skip (model, partition) already present
        if test_path.exists():
            with open(test_path) as f:
                for row in csv.DictReader(f):
                    results_ts0.append({k: row.get(k, '') for k in fieldnames})
        if new_path.exists():
            with open(new_path) as f:
                for row in csv.DictReader(f):
                    results_new.append({k: row.get(k, '') for k in fieldnames})
        have_ts0 = {r['model'] for r in results_ts0}
        have_new = {r['model'] for r in results_new}
        to_ts0 = [m for m in args.models if m in config and m not in have_ts0]
        to_new = [m for m in args.models if m in config and m not in have_new]
        log(f"Resume: TS0 done {have_ts0}, NEW done {have_new}")
        log(f"Resume: will compute TS0 for {to_ts0}, NEW for {to_new}")
    elif args.dataset == 'new':
        if test_path.exists():
            with open(test_path) as f:
                for row in csv.DictReader(f):
                    results_ts0.append({k: row.get(k, '') for k in fieldnames})
        if new_path.exists():
            with open(new_path) as f:
                for row in csv.DictReader(f):
                    if row.get('model') not in args.models:
                        results_new.append({k: row.get(k, '') for k in fieldnames})
        log(f"Dataset=new: loaded TS0 from {test_path}, will compute NEW only for {args.models}")
    elif args.dataset == 'ts0':
        if new_path.exists():
            with open(new_path) as f:
                for row in csv.DictReader(f):
                    results_new.append({k: row.get(k, '') for k in fieldnames})
        log(f"Dataset=ts0: loaded NEW from {new_path} (will preserve), will compute TS0 only")
    elif args.overwrite:
        models_to_overwrite = set(args.models)
        for p, res_list in [(test_path, results_ts0), (new_path, results_new)]:
            if p.exists():
                with open(p) as f:
                    for row in csv.DictReader(f):
                        if row.get('model') not in models_to_overwrite:
                            res_list.append({k: row.get(k, '') for k in fieldnames})
        log(f"Overwrite mode: loaded {len(results_ts0)} TS0 rows, {len(results_new)} NEW rows (will replace {args.models})")

    def write_partial_csv():
        test_path = out_dir / 'final_test_metrics.csv'
        with open(test_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results_ts0:
                w.writerow({k: r[k] for k in fieldnames})
        new_path = out_dir / 'final_new_metrics.csv'
        with open(new_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results_new:
                w.writerow({k: r[k] for k in fieldnames})

    models_to_run = args.models
    if args.resume:
        models_to_run = [m for m in args.models if m in config and (m in to_ts0 or m in to_new)]
        if not models_to_run:
            log("Resume: nothing to do, all done.")
            return

    for model in models_to_run:
        cfg = config.get(model)
        if not cfg:
            continue
        need_ts0 = (args.dataset in ('ts0', 'both') and (not args.resume or model in to_ts0))
        need_new = (args.dataset in ('new', 'both') and (not args.resume or model in to_new))
        if not need_ts0 and not need_new:
            continue
        ckpt = Path(args.checkpoint_base) / model / f"layer_{cfg['layer']}" / f"k_{cfg['k']}" / "seed_42" / "best.pt"
        if not ckpt.exists():
            log(f"[WARN] Checkpoint not found: {ckpt}")
            continue
        B, k, d = load_probe_matrix(str(ckpt))
        B = B.cpu()
        thresh = cfg['threshold']
        decoding_mode = cfg['decoding_mode']

        # TS0
        if need_ts0:
            tp_sum, fp_sum, fn_sum = 0, 0, 0
            cr_sum, cr_denom = 0.0, 0
            n_ts0 = 0
            for idx, seq_id in enumerate(ts0_ids):
                if seq_id not in bpRNA:
                    continue
                seq = bpRNA[seq_id]['sequence']
                true_pairs = parse_base_pairs(bpRNA[seq_id]['base_pairs'])
                emb_path = get_embedding_path(seq_id, model, cfg['layer'], args.embeddings_base)
                res = eval_one_seq(seq_id, seq, true_pairs, emb_path, B, thresh, decoding_mode, model=model)
                if res is not None:
                    tp_sum += res['TP']
                    fp_sum += res['FP']
                    fn_sum += res['FN']
                    pred_count = res['TP'] + res['FP']
                    if pred_count > 0 and not np.isnan(res['canonical_rate']):
                        cr_sum += res['canonical_rate'] * pred_count
                        cr_denom += pred_count
                    n_ts0 += 1
                if (idx + 1) % 300 == 0:
                    log(f"{model} TS0: {idx+1}/{len(ts0_ids)}")
            if n_ts0 > 0:
                prec, rec, f1 = precision_recall_f1(tp_sum, fp_sum, fn_sum)
                cr = cr_sum / cr_denom if cr_denom > 0 else (1.0 if decoding_mode in ('canonical_wobble', 'canonical_constrained') else 0.0)
                results_ts0.append({
                    'model': model, 'layer': cfg['layer'], 'k': cfg['k'], 'seed': 42,
                    'threshold': thresh, 'decoding_mode': decoding_mode,
                    'precision': prec, 'recall': rec, 'f1': f1,
                    'TP': tp_sum, 'FP': fp_sum, 'FN': fn_sum, 'canonical_rate': cr,
                })
                log(f"{model} TS0: F1={f1:.4f} (n={n_ts0})")
            write_partial_csv()

        # NEW
        if need_new:
            tp_sum, fp_sum, fn_sum = 0, 0, 0
            cr_sum, cr_denom = 0.0, 0
            n_new = 0
            for idx, seq_id in enumerate(new_ids):
                if seq_id not in bpRNA:
                    continue
                seq = bpRNA[seq_id]['sequence']
                true_pairs = parse_base_pairs(bpRNA[seq_id]['base_pairs'])
                emb_path = get_embedding_path(seq_id, model, cfg['layer'], args.embeddings_base)
                res = eval_one_seq(seq_id, seq, true_pairs, emb_path, B, thresh, decoding_mode, model=model)
                if res is not None:
                    tp_sum += res['TP']
                    fp_sum += res['FP']
                    fn_sum += res['FN']
                    pred_count = res['TP'] + res['FP']
                    if pred_count > 0 and not np.isnan(res['canonical_rate']):
                        cr_sum += res['canonical_rate'] * pred_count
                        cr_denom += pred_count
                    n_new += 1
                if (idx + 1) % 500 == 0:
                    log(f"{model} NEW: {idx+1}/{len(new_ids)}")
            if n_new > 0:
                prec, rec, f1 = precision_recall_f1(tp_sum, fp_sum, fn_sum)
                cr = cr_sum / cr_denom if cr_denom > 0 else (1.0 if decoding_mode in ('canonical_wobble', 'canonical_constrained') else 0.0)
                results_new.append({
                    'model': model, 'layer': cfg['layer'], 'k': cfg['k'], 'seed': 42,
                    'threshold': thresh, 'decoding_mode': decoding_mode,
                    'precision': prec, 'recall': rec, 'f1': f1,
                    'TP': tp_sum, 'FP': fp_sum, 'FN': fn_sum, 'canonical_rate': cr,
                })
                log(f"{model} NEW: F1={f1:.4f} (n={n_new})")
        write_partial_csv()

    log(f"Done. Wrote {out_dir / 'final_test_metrics.csv'} and {out_dir / 'final_new_metrics.csv'}")


if __name__ == '__main__':
    main()
