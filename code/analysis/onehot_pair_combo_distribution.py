#!/usr/bin/env python3
"""
Unconstrained: 각 16가지 염기쌍 조합별 예측 비율 (count, rate).
지원 모델: onehot, rinalmo (--model 옵션)
Output: {model}_pair_combo_distribution.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/projects/u6cg/jay/dissertations/jan22')
from utils.evaluation import prob_to_pairs
from scripts.generation.generate_base_pairs import load_probe_matrix

BASES = ['A', 'U', 'G', 'C']
MODEL_CONFIG = {
    'onehot': {'layer': 0, 'k': 128, 'thresh': 0.65, 'dir': 'Onehot'},
    'rinalmo': {'layer': 10, 'k': 32, 'thresh': 0.7, 'dir': 'RiNALMo'},
}


def get_embedding_path(seq_id, model, layer):
    base = Path('/projects/u6cg/jay/dissertations/data/embeddings')
    model_dir = MODEL_CONFIG.get(model, {}).get('dir', model)
    return base / model_dir / 'bpRNA' / 'by_layer' / f'layer_{layer}' / f'{seq_id}.npy'


def run_model(model_name, seq_ids, bpRNA, limit=None):
    cfg = MODEL_CONFIG.get(model_name)
    if not cfg:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIG)}")
    layer, k, thresh = cfg['layer'], cfg['k'], cfg['thresh']
    ckpt = Path(f'/projects/u6cg/jay/dissertations/feb8/results_updated/outputs/{model_name}/layer_{layer}/k_{k}/seed_42/best.pt')
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    B, _, _ = load_probe_matrix(str(ckpt))
    B = B.cpu()

    combo_counts = {(b1, b2): 0 for b1 in BASES for b2 in BASES}
    total_pred = 0
    ids = seq_ids[:limit] if limit else seq_ids

    for seq_id in ids:
        if seq_id not in bpRNA:
            continue
        seq = bpRNA[seq_id]
        emb_path = get_embedding_path(seq_id, model_name, layer)
        if not emb_path.exists():
            continue
        emb = np.load(emb_path)
        emb_t = torch.from_numpy(emb).float()
        z = torch.matmul(emb_t, B.t())
        p_ij = torch.sigmoid(torch.matmul(z, z.t()))
        pred_pairs = prob_to_pairs(p_ij, threshold=thresh, allowed_mask=None, inputs_are_logits=False)
        for i, j in pred_pairs:
            bi = seq[i - 1]
            bj = seq[j - 1]
            if bi in BASES and bj in BASES:
                combo_counts[(bi, bj)] += 1
                total_pred += 1

    return combo_counts, total_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='onehot', choices=['onehot', 'rinalmo'], help='Model to analyze')
    ap.add_argument('--limit', type=int, default=None, help='Limit sequences (for quick test)')
    args = ap.parse_args()

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

    combo_counts, total_pred = run_model(args.model, seq_ids, bpRNA, args.limit)

    out_path = Path(f'/projects/u6cg/jay/dissertations/march1/data/{args.model}_pair_combo_distribution.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for b1 in BASES:
        for b2 in BASES:
            cnt = combo_counts[(b1, b2)]
            rate = cnt / total_pred if total_pred else 0
            wc_gu = (b1, b2) in {('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')}
            rows.append({
                'pair': f'{b1}{b2}',
                'count': cnt,
                'rate': rate,
                'rate_pct': f'{100*rate:.2f}%',
                'wc_gu': 'yes' if wc_gu else 'no',
            })

    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair', 'count', 'rate', 'rate_pct', 'wc_gu'])
        w.writeheader()
        w.writerows(rows)

    print(f"[{args.model}] Total predicted pairs: {total_pred}")
    print(f"Saved {out_path}")
    print("\n--- 16 combinations (ordered) ---")
    for r in rows:
        print(f"  {r['pair']}: {r['count']:>8} ({r['rate_pct']:>6}) {r['wc_gu']}")
    wc_gu_total = sum(combo_counts[(b1,b2)] for b1 in BASES for b2 in BASES
                     if (b1,b2) in {('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')})
    print(f"\nWC+GU total: {wc_gu_total} ({100*wc_gu_total/total_pred:.2f}%)" if total_pred else "")


if __name__ == '__main__':
    main()
