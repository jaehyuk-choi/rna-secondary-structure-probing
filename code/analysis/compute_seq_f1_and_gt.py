#!/usr/bin/env python3
"""Per-seq F1 + GT pairs for heatmap IDs (uses best_config_val_f1)."""

import ast
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.evaluation import prob_to_pairs, create_canonical_mask
from probe_inference.generate_base_pairs import load_probe_matrix

DATA = REPO_ROOT / 'data'
BEST_CONFIG = REPO_ROOT / 'configs' / 'best_config_val_f1.csv'

MODEL_EMBED = {'ernie': 'RNAErnie', 'roberta': 'RoBERTa', 'rnabert': 'RNABERT', 'rnafm': 'RNAFM', 'onehot': 'Onehot'}
MODEL_LABELS = {'ernie': 'ERNIE-RNA', 'roberta': 'RoBERTa', 'rnabert': 'RNABERT', 'rnafm': 'RNAFM', 'onehot': 'One-hot'}
MODELS = ['ernie', 'roberta', 'rnabert', 'rnafm', 'onehot']


def load_best_config():
    cfg = {}
    with open(BEST_CONFIG) as f:
        for r in csv.DictReader(f):
            m = r['model']
            if m == 'model' or m not in MODELS:
                continue
            cfg[m] = {
                'layer': int(r['selected_layer']),
                'k': int(r['selected_k']),
                'threshold': float(r['selected_best_threshold']),
                'decoding_mode': r['selected_decoding_mode'],
            }
    return cfg


def get_embedding_path(seq_id, model, layer):
    return DATA / 'embeddings' / MODEL_EMBED[model] / 'bpRNA' / 'by_layer' / f'layer_{layer}' / f'{seq_id}.npy'


def get_probe_path(model, layer, k):
    return REPO_ROOT / 'results' / 'outputs' / model / f'layer_{layer}' / f'k_{k}' / 'seed_42' / 'best.pt'


def compute_P(emb, B):
    H = torch.from_numpy(emb).float()
    z = torch.matmul(H, B.t())
    s = torch.matmul(z, z.t())
    return torch.sigmoid(s)


def compute_f1_with_matches(pred_pairs, true_pairs, shift=1):
    """Return F1, TP, FP, FN and also TP_pairs, FP_pairs, FN_pairs for comparison."""
    TP_pairs = []  # (pred, matched_gt)
    FP_pairs = []
    used_gt = set()

    for (pi, pj) in pred_pairs:
        matched = False
        for (ti, tj) in true_pairs:
            if (abs(pi - ti) <= shift and pj == tj) or (pi == ti and abs(pj - tj) <= shift):
                TP_pairs.append(((pi, pj), (ti, tj)))
                used_gt.add((ti, tj))
                matched = True
                break
        if not matched:
            FP_pairs.append((pi, pj))

    FN_pairs = [(ti, tj) for (ti, tj) in true_pairs if (ti, tj) not in used_gt]

    TP, FP, FN = len(TP_pairs), len(FP_pairs), len(FN_pairs)
    P = TP / (TP + FP) if (TP + FP) > 0 else 0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    return F1, TP, FP, FN, TP_pairs, FP_pairs, FN_pairs


def main():
    bpRNA = {}
    with open(DATA / 'metadata' / 'bpRNA.csv') as f:
        for r in csv.DictReader(f):
            bpRNA[r['id']] = {'seq': r['sequence'], 'pairs': r['base_pairs']}

    best_cfg = load_best_config()
    seq_ids = ['bpRNA_RFAM_22136', 'bpRNA_RFAM_21498']

    for seq_id in seq_ids:
        if seq_id not in bpRNA:
            continue
        seq = bpRNA[seq_id]['seq']
        true_pairs = ast.literal_eval(bpRNA[seq_id]['pairs'])
        L = len(seq)

        print("=" * 70)
        print(f"Sequence: {seq_id}  |  L = {L}  |  Ground truth pairs: {len(true_pairs)}")
        print("=" * 70)
        print("\n[Ground truth base pairs (1-based, i < j)]:")
        for i in range(0, len(true_pairs), 5):
            row = true_pairs[i : i + 5]
            print("  " + "  ".join(f"({a},{b})" for a, b in row))
        print()

        for model in MODELS:
            if model not in best_cfg:
                continue
            cfg = best_cfg[model]
            layer, k, tau, mode = cfg['layer'], cfg['k'], cfg['threshold'], cfg['decoding_mode']

            emb_path = get_embedding_path(seq_id, model, layer)
            ckpt_path = get_probe_path(model, layer, k)
            if not emb_path.exists() or not ckpt_path.exists():
                print(f"  warn: {model}: missing emb or ckpt")
                continue

            emb = np.load(emb_path)
            B, _, _ = load_probe_matrix(str(ckpt_path))
            B = B.cpu()
            P = compute_P(emb, B)

            if mode == 'unconstrained':
                allowed = None
            else:
                allowed = create_canonical_mask(seq, allow_gu=(mode == 'canonical_wobble'))

            pred_pairs = prob_to_pairs(P, threshold=tau, allowed_mask=allowed, inputs_are_logits=False)
            F1, TP, FP, FN, TP_pairs, FP_pairs, FN_pairs = compute_f1_with_matches(pred_pairs, true_pairs)

            print(f"  {MODEL_LABELS[model]:12} | τ={tau} | F1={F1:.3f} | TP={TP} FP={FP} FN={FN} | pred={len(pred_pairs)}")
            print(f"      TP (matched): {[p for p, g in TP_pairs]}")
            print(f"      FP (incorrect): {FP_pairs}")
            print(f"      FN (missed): {FN_pairs}")
            print()
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
