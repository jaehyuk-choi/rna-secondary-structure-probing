#!/usr/bin/env python3
"""Generate base_pair.txt files for VL0 (validation) sequences.

Same logic as feb8/validation_based_optimal/generate_vl0_base_pairs.py, but:
- Writes under feb23
- Includes rnabert -> RNABERT embedding dir mapping
- Applies decoding_mode (canonical_constrained, canonical_wobble, unconstrained) so
  bonus pairs match probe-only evaluation (only canonical pairs when configured).
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add jan22 for generate_base_pairs and evaluation utils
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'jan22'))
from scripts.generation.generate_base_pairs import load_probe_matrix
from utils.evaluation import create_canonical_mask


def get_embedding_path(seq_id, model, layer, embeddings_base):
    emb_base = Path(embeddings_base)
    model_dirs = {
        'ernie': 'RNAErnie',
        'roberta': 'RoBERTa',
        'rnafm': 'RNAFM',
        'rinalmo': 'RiNALMo',
        'onehot': 'Onehot',
        'rnabert': 'RNABERT',
    }
    model_dir = model_dirs.get(model, model)
    layer_path = emb_base / model_dir / 'bpRNA' / 'by_layer' / f'layer_{layer}'
    return layer_path / f'{seq_id}.npy'


def _process_one_seq(args):
    seq_id, emb_path, out_path, B_npy, thresh, sequence, allow_gu, force, use_threshold, model = args
    import numpy as np
    import torch

    if not Path(emb_path).exists():
        return seq_id, False, 'no embedding'
    if Path(out_path).exists() and not force:
        return seq_id, True, 'exists'

    try:
        B = torch.from_numpy(B_npy).float()
        emb = np.load(emb_path)
        emb_t = torch.from_numpy(emb).float()
        z = torch.matmul(emb_t, B.t())
        p_ij = torch.sigmoid(torch.matmul(z, z.t()))
        # RNAFM adds BOS/EOS at END: emb has seq_len+2; bases are first L, use [0:-2, 0:-2]
        if sequence and p_ij.shape[0] == len(sequence) + 2 and model == 'rnafm':
            p_ij = p_ij[0:-2, 0:-2]
        L = p_ij.shape[0]

        # Apply decoding_mode: only include pairs allowed by canonical mask
        # When use_threshold=False, write all pairs (like TS0/NEW) for CPLfold bonus
        thresh_ok = (lambda s: s >= thresh) if use_threshold else (lambda s: True)
        if allow_gu is not None and sequence:
            mask = create_canonical_mask(sequence, allow_gu=allow_gu, device='cpu')
            pairs = [
                (i + 1, j + 1, p_ij[i, j].item())
                for i in range(L)
                for j in range(i + 1, L)
                if thresh_ok(p_ij[i, j].item()) and mask[i, j].item()
            ]
        else:
            pairs = [
                (i + 1, j + 1, p_ij[i, j].item())
                for i in range(L)
                for j in range(i + 1, L)
                if thresh_ok(p_ij[i, j].item())
            ]

        with open(out_path, 'w') as f:
            for i, j, s in pairs:
                f.write(f"{i}\t{j}\t{s:.6f}\n")
        return seq_id, True, 'ok'
    except Exception as e:
        return seq_id, False, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits-csv', default='/projects/u6cg/jay/dissertations/data/bpRNA_splits.csv')
    ap.add_argument('--bpRNA-csv', default='/projects/u6cg/jay/dissertations/data/bpRNA.csv')
    ap.add_argument('--config-csv', default='/projects/u6cg/jay/dissertations/jan22/results_updated/summary/final_selected_config.csv')
    ap.add_argument('--embeddings-base', default='/projects/u6cg/jay/dissertations/data/embeddings')
    ap.add_argument('--checkpoint-base', default='/projects/u6cg/jay/dissertations/jan22/results_updated/outputs')
    ap.add_argument('--output-dir', default='/projects/u6cg/jay/dissertations/feb23/vl0_base_pairs')
    ap.add_argument('--models', nargs='+', default=['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert'])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--num-workers', type=int, default=20)
    ap.add_argument('--force', action='store_true', help='Overwrite existing base_pair files')
    ap.add_argument('--no-threshold', action='store_true', help='Write all pairs (no threshold filter), like TS0/NEW')
    args = ap.parse_args()

    use_threshold = not args.no_threshold

    vl0_ids = []
    with open(args.splits_csv) as f:
        for row in csv.DictReader(f):
            if row.get('partition', '').strip().upper() == 'VL0':
                vl0_ids.append(row['id'])
    print(f"[INFO] Found {len(vl0_ids)} VL0 sequences")

    config = {}
    with open(args.config_csv) as f:
        for row in csv.DictReader(f):
            m = row['model']
            if m in args.models:
                config[m] = {
                    'layer': int(row['selected_layer']),
                    'k': int(row['selected_k']),
                    'threshold': float(row['selected_best_threshold']),
                    'decoding_mode': row.get('selected_decoding_mode', 'unconstrained'),
                }

    # Load sequences for decoding_mode (canonical mask)
    bpRNA = {}
    with open(args.bpRNA_csv) as f:
        for row in csv.DictReader(f):
            bpRNA[row['id']] = row.get('sequence', row.get('seq', ''))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        (out_dir / model).mkdir(exist_ok=True)
        cfg = config.get(model)
        if not cfg:
            print(f"[WARN] No config for model={model}")
            continue

        ckpt = (
            Path(args.checkpoint_base)
            / model
            / f"layer_{cfg['layer']}"
            / f"k_{cfg['k']}"
            / 'seed_42'
            / 'best.pt'
        )
        if not ckpt.exists():
            print(f"[WARNING] Checkpoint not found: {ckpt}")
            continue

        B, k, d = load_probe_matrix(str(ckpt))
        thresh = cfg['threshold']
        decoding_mode = cfg.get('decoding_mode', 'unconstrained')
        allow_gu = None if decoding_mode == 'unconstrained' else (decoding_mode == 'canonical_wobble')
        B_npy = B.numpy()

        seqs = vl0_ids[: args.limit] if args.limit else vl0_ids
        tasks = []
        for seq_id in seqs:
            emb_path = get_embedding_path(seq_id, model, cfg['layer'], args.embeddings_base)
            out_path = out_dir / model / f'base_pair_{model}_{seq_id}.txt'
            sequence = bpRNA.get(seq_id, '')
            tasks.append((seq_id, str(emb_path), str(out_path), B_npy, thresh, sequence, allow_gu, args.force, use_threshold, model))

        n_workers = min(args.num_workers, len(tasks))
        print(f"[INFO] {model}: {len(tasks)} sequences, {n_workers} workers, decoding_mode={decoding_mode}")

        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for fut in as_completed(ex.submit(_process_one_seq, t) for t in tasks):
                sid, ok, msg = fut.result()
                if not ok and msg != 'exists':
                    print(f"[WARN] {model} {sid}: {msg}")
                done += 1
                if done % 200 == 0:
                    print(f"[INFO] {model}: {done}/{len(tasks)}")

    print(f"[INFO] Done. Output: {out_dir}")


if __name__ == '__main__':
    raise SystemExit(main())
