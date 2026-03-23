#!/usr/bin/env python3
"""
Probe-only metrics with `canonical_rate_wobble`, which includes GU and UG pairs.

This script follows the same evaluation logic as `compute_feb8_probe_only_metrics.py`
but adds `canonical_rate_wobble`.
Outputs:
- `final_test_metrics_wobble.csv`
- `final_new_metrics_wobble.csv`

The standard output files are left unchanged.
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
    """Compute F1 and both canonical_rate (no GU) and canonical_rate_wobble (with GU/UG)."""
    if not Path(emb_path).exists():
        return None
    try:
        emb = np.load(emb_path)
        emb_t = torch.from_numpy(emb).float().to(device)
        z = torch.matmul(emb_t, B.t())
        p_ij = torch.sigmoid(torch.matmul(z, z.t()))
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
        canonical_rate_wobble = float('nan')
        if pred_pairs and sequence:
            canonical_rate = compute_canonical_rate(pred_pairs, sequence, allow_gu=False)
            canonical_rate_wobble = compute_canonical_rate(pred_pairs, sequence, allow_gu=True)
        return {
            'TP': TP, 'FP': FP, 'FN': FN, 'precision': prec, 'recall': rec, 'f1': f1,
            'canonical_rate': canonical_rate, 'canonical_rate_wobble': canonical_rate_wobble,
        }
    except Exception as e:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-csv', default='/projects/u6cg/jay/dissertations/march1/final_selected_config_unconstrained.csv')
    ap.add_argument('--splits-csv', default='/projects/u6cg/jay/dissertations/data/bpRNA_splits.csv')
    ap.add_argument('--bpRNA-csv', default='/projects/u6cg/jay/dissertations/data/bpRNA.csv')
    ap.add_argument('--embeddings-base', default='/projects/u6cg/jay/dissertations/data/embeddings')
    ap.add_argument('--checkpoint-base', default='/projects/u6cg/jay/dissertations/feb8/results_updated/outputs')
    ap.add_argument('--output-dir', default='/projects/u6cg/jay/dissertations/march1')
    ap.add_argument('--progress-log', default='', help='Path to progress log')
    ap.add_argument('--models', nargs='+', default=['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert'])
    args = ap.parse_args()

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

    bpRNA = {}
    with open(args.bpRNA_csv) as f:
        for row in csv.DictReader(f):
            bpRNA[row['id']] = {'sequence': row['sequence'], 'base_pairs': row['base_pairs']}

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
    progress_log = Path(args.progress_log) if args.progress_log else out_dir / 'probe_only_wobble_progress.log'

    def log(msg):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if progress_log:
            with open(progress_log, 'a') as f:
                f.write(line + '\n')
                f.flush()

    log(f"Probe-only with wobble. TS0: {len(ts0_ids)}, NEW: {len(new_ids)}")
    log(f"Output: final_test_metrics_wobble.csv, final_new_metrics_wobble.csv")

    fieldnames = ['model', 'layer', 'k', 'seed', 'threshold', 'decoding_mode',
                  'precision', 'recall', 'f1', 'TP', 'FP', 'FN',
                  'canonical_rate', 'canonical_rate_wobble']

    results_ts0 = []
    results_new = []

    for model in args.models:
        cfg = config.get(model)
        if not cfg:
            continue
        ckpt = Path(args.checkpoint_base) / model / f"layer_{cfg['layer']}" / f"k_{cfg['k']}" / "seed_42" / "best.pt"
        if not ckpt.exists():
            log(f"[WARN] Checkpoint not found: {ckpt}")
            continue
        B, k, d = load_probe_matrix(str(ckpt))
        B = B.cpu()
        thresh = cfg['threshold']
        decoding_mode = cfg['decoding_mode']

        tp_sum, fp_sum, fn_sum = 0, 0, 0
        cr_sum, cr_denom = 0.0, 0
        crw_sum, crw_denom = 0.0, 0
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
                if pred_count > 0 and not np.isnan(res['canonical_rate_wobble']):
                    crw_sum += res['canonical_rate_wobble'] * pred_count
                    crw_denom += pred_count
                n_ts0 += 1
            if (idx + 1) % 300 == 0:
                log(f"{model} TS0: {idx+1}/{len(ts0_ids)}")
        if n_ts0 > 0:
            prec, rec, f1 = precision_recall_f1(tp_sum, fp_sum, fn_sum)
            cr = cr_sum / cr_denom if cr_denom > 0 else 0.0
            crw = crw_sum / crw_denom if crw_denom > 0 else 0.0
            results_ts0.append({
                'model': model, 'layer': cfg['layer'], 'k': cfg['k'], 'seed': 42,
                'threshold': thresh, 'decoding_mode': decoding_mode,
                'precision': prec, 'recall': rec, 'f1': f1,
                'TP': tp_sum, 'FP': fp_sum, 'FN': fn_sum,
                'canonical_rate': cr, 'canonical_rate_wobble': crw,
            })
            log(f"{model} TS0: F1={f1:.4f} cr={cr:.3f} cr_wobble={crw:.3f} (n={n_ts0})")

        tp_sum, fp_sum, fn_sum = 0, 0, 0
        cr_sum, cr_denom = 0.0, 0
        crw_sum, crw_denom = 0.0, 0
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
                if pred_count > 0 and not np.isnan(res['canonical_rate_wobble']):
                    crw_sum += res['canonical_rate_wobble'] * pred_count
                    crw_denom += pred_count
                n_new += 1
            if (idx + 1) % 500 == 0:
                log(f"{model} NEW: {idx+1}/{len(new_ids)}")
        if n_new > 0:
            prec, rec, f1 = precision_recall_f1(tp_sum, fp_sum, fn_sum)
            cr = cr_sum / cr_denom if cr_denom > 0 else 0.0
            crw = crw_sum / crw_denom if crw_denom > 0 else 0.0
            results_new.append({
                'model': model, 'layer': cfg['layer'], 'k': cfg['k'], 'seed': 42,
                'threshold': thresh, 'decoding_mode': decoding_mode,
                'precision': prec, 'recall': rec, 'f1': f1,
                'TP': tp_sum, 'FP': fp_sum, 'FN': fn_sum,
                'canonical_rate': cr, 'canonical_rate_wobble': crw,
            })
            log(f"{model} NEW: F1={f1:.4f} cr={cr:.3f} cr_wobble={crw:.3f} (n={n_new})")

        with open(out_dir / 'final_test_metrics_wobble.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results_ts0)
        with open(out_dir / 'final_new_metrics_wobble.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results_new)

    log(f"Done. Wrote {out_dir / 'final_test_metrics_wobble.csv'} and {out_dir / 'final_new_metrics_wobble.csv'}")


if __name__ == '__main__':
    main()
