#!/usr/bin/env python3
"""
Structural probing contact map visualization.

For selected RNA sequences from TS0, generates side-by-side heatmaps:
- Left: Ground truth binary contact map (binary, upper triangle)
- Right: Predicted score matrix P = sigmoid(z_i^T z_j) (continuous 0-1, "hot" colormap)
- Optional: Third panel with overlay (--overlay): GT in blue, pred@τ in red

Uses validation-selected best config (layer, k, τ) per model.
Saves PDF and PNG at 300 DPI to figures/probe_heatmaps/.

Requires: conda activate rna_probe (torch, matplotlib)

Usage:
  python scripts/plot_probe_contact_heatmaps.py --n-seq 3
  python scripts/plot_probe_contact_heatmaps.py --seq-ids bpRNA_RFAM_22136 bpRNA_RFAM_21498 --overlay
"""

import argparse
import ast
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/projects/u6cg/jay/dissertations/jan22')
from scripts.generation.generate_base_pairs import load_probe_matrix
from utils.evaluation import prob_to_pairs, create_canonical_mask

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

MARCH1 = Path(__file__).resolve().parents[1]
DATA = Path('/projects/u6cg/jay/dissertations/data')
FEB8 = Path('/projects/u6cg/jay/dissertations/feb8/results_updated')
BEST_CONFIG = FEB8 / 'summary/final_selected_config.csv'

# Model -> (embedding_dir, layer from best config)
MODEL_EMBED = {
    'ernie': 'RNAErnie',
    'roberta': 'RoBERTa',
    'rnabert': 'RNABERT',
    'rnafm': 'RNAFM',
    'rinalmo': 'RiNALMo',
    'onehot': 'Onehot',
}
MODEL_LABELS = {
    'ernie': 'ERNIE-RNA',
    'roberta': 'RoBERTa',
    'rnabert': 'RNABERT',
    'rnafm': 'RNAFM',
    'rinalmo': 'RiNALMo',
    'onehot': 'One-hot',
}
MODELS = ['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert']  # Ordered by probe-only performance.


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


def compute_tp_fp_fn(pred_pairs, true_pairs, shift=1):
    """Return (TP_pairs, FP_pairs, FN_pairs)."""
    TP_pairs, FP_pairs = [], []
    used_gt = set()
    for (pi, pj) in pred_pairs:
        matched = False
        for (ti, tj) in true_pairs:
            if (abs(pi - ti) <= shift and pj == tj) or (pi == ti and abs(pj - tj) <= shift):
                TP_pairs.append((pi, pj))
                used_gt.add((ti, tj))
                matched = True
                break
        if not matched:
            FP_pairs.append((pi, pj))
    FN_pairs = [(ti, tj) for (ti, tj) in true_pairs if (ti, tj) not in used_gt]
    return TP_pairs, FP_pairs, FN_pairs


def get_embedding_path(seq_id, model, layer):
    dir_name = MODEL_EMBED[model]
    return DATA / 'embeddings' / dir_name / 'bpRNA' / 'by_layer' / f'layer_{layer}' / f'{seq_id}.npy'


def get_probe_path(model, layer, k):
    return FEB8 / 'outputs' / model / f'layer_{layer}' / f'k_{k}' / 'seed_42' / 'best.pt'


def pairs_to_contact_map(pairs, L):
    """Convert list of (i,j) 1-based pairs to L×L binary contact map (upper triangle)."""
    M = np.zeros((L, L), dtype=np.float32)
    for i, j in pairs:
        if 1 <= i <= L and 1 <= j <= L and i < j:
            M[i - 1, j - 1] = 1
    return M


def compute_P(embeddings, B):
    """P[i,j] = sigmoid(z_i^T z_j), upper triangle only for display."""
    H = torch.from_numpy(embeddings).float()
    z = torch.matmul(H, B.t())  # (L, k)
    s = torch.matmul(z, z.t())  # (L, L)
    P = torch.sigmoid(s).numpy()
    return P


def _run_seq_figure(seq_id, out_dir, best_cfg, bpRNA, model_filter=None, suffix=''):
    """One figure: models as rows, 4 cols (GT, Predicted, Pred≥τ, TP/FP/FN).
    model_filter: list of model keys to include (None = all MODELS).
    suffix: appended to output filename, e.g. '_selected'.
    """
    if seq_id not in bpRNA:
        print(f"[ERROR] {seq_id} not in bpRNA")
        return 1
    seq = bpRNA[seq_id]['seq']
    pairs = ast.literal_eval(bpRNA[seq_id]['pairs'])
    L = len(seq)
    M_gt = pairs_to_contact_map(pairs, L)
    M_gt_ut = np.triu(M_gt, 1) + np.tril(np.nan * np.ones_like(M_gt))

    models_to_plot = model_filter if model_filter else MODELS
    n_models = len(models_to_plot)
    n_cols = 4  # GT | Predicted | Pred ≥ τ | TP/FP/FN
    COL_TITLES = ['Ground Truth', 'Predicted', None, None]  # col 2,3 set per-model

    fig, axes = plt.subplots(n_models, n_cols,
                             figsize=(5.5 * n_cols, 5 * n_models), squeeze=False)

    print(f"  Sequence: {seq_id}  L={L}  GT pairs={len(pairs)}")

    for m_idx, model in enumerate(models_to_plot):
        if model not in best_cfg:
            continue
        cfg = best_cfg[model]
        layer, k, tau = cfg['layer'], cfg['k'], cfg['threshold']
        mode = cfg.get('decoding_mode', 'unconstrained')

        emb_path = get_embedding_path(seq_id, model, layer)
        ckpt_path = get_probe_path(model, layer, k)
        if not emb_path.exists() or not ckpt_path.exists():
            print(f"  [WARN] Missing data for {model}")
            continue

        emb = np.load(emb_path)
        B, _, _ = load_probe_matrix(str(ckpt_path))
        B = B.cpu()
        P = compute_P(emb, B)
        P_torch = torch.from_numpy(P).float()
        P_ut = np.triu(P, 1) + np.tril(np.nan * np.ones_like(P))

        if mode == 'unconstrained':
            allowed = None
        else:
            allowed = create_canonical_mask(seq, allow_gu=(mode == 'canonical_wobble'))
        pred_pairs = prob_to_pairs(P_torch, threshold=tau, allowed_mask=allowed, inputs_are_logits=False)
        TP_pairs, FP_pairs, FN_pairs = compute_tp_fp_fn(pred_pairs, pairs)

        model_label = MODEL_LABELS[model]

        from matplotlib.patches import Patch

        for c in range(n_cols):
            axes[m_idx, c].set_xlabel('Nucleotide position', fontsize=10)
            if c > 0:
                axes[m_idx, c].set_ylabel('Nucleotide position', fontsize=10)

        # Col 0: Ground truth contact map
        ax0 = axes[m_idx, 0]
        ax0.imshow(M_gt_ut, cmap='Blues', aspect='equal', vmin=0, vmax=1)
        ax0.set_xlim(-0.5, L - 0.5)
        ax0.set_ylim(L - 0.5, -0.5)
        if m_idx == 0:
            ax0.set_title('Ground-Truth\nContact Map', fontsize=14, fontweight='bold', pad=14)
        ax0.set_ylabel('Nucleotide position', fontsize=10, labelpad=8)
        ax0.annotate(model_label, xy=(0, 0.5), xytext=(-50, 0),
                     xycoords='axes fraction', textcoords='offset points',
                     ha='right', va='center', fontsize=14, fontweight='bold', rotation=90)
        ax0.tick_params(labelsize=9)

        # Col 1: Predicted pair probability
        ax1 = axes[m_idx, 1]
        im1 = ax1.imshow(P_ut, cmap='hot', aspect='equal', vmin=0, vmax=1)
        ax1.set_xlim(-0.5, L - 0.5)
        ax1.set_ylim(L - 0.5, -0.5)
        if m_idx == 0:
            ax1.set_title('Pairwise Probabilities\n$p_{ij}$', fontsize=14, fontweight='bold', pad=14)
        ax1.tick_params(labelsize=9)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='$p_{ij}$')

        # Col 2: Thresholded prediction — τ as legend box in lower left
        ax2 = axes[m_idx, 2]
        im2 = ax2.imshow(P_ut, cmap='hot', aspect='equal', vmin=tau, vmax=1)
        ax2.set_xlim(-0.5, L - 0.5)
        ax2.set_ylim(L - 0.5, -0.5)
        if m_idx == 0:
            ax2.set_title('Thresholded Map\n' + r'($p_{ij} \geq \tau$)',
                          fontsize=14, fontweight='bold', pad=14)
        tau_patch = Patch(facecolor='none', edgecolor='none', label=f'$\\tau$ = {tau:.2f}')
        ax2.legend(handles=[tau_patch], loc='lower left', fontsize=10,
                   framealpha=0.9, edgecolor='gray', handlelength=0, handletextpad=0)
        ax2.tick_params(labelsize=9)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='$p_{ij}$')

        # Col 3: TP/FP/FN (upper triangle colored, lower triangle white)
        ax3 = axes[m_idx, 3]
        rgb = np.ones((L, L, 3))  # white background
        for (i, j) in TP_pairs:
            if 0 <= i - 1 < L and 0 <= j - 1 < L and i < j:
                rgb[i - 1, j - 1] = [0.2, 0.8, 0.2]
        for (i, j) in FP_pairs:
            if 0 <= i - 1 < L and 0 <= j - 1 < L and i < j:
                rgb[i - 1, j - 1] = [0.9, 0.2, 0.2]
        for (i, j) in FN_pairs:
            if 0 <= i - 1 < L and 0 <= j - 1 < L and i < j:
                rgb[i - 1, j - 1] = [0.2, 0.2, 0.9]
        for ii in range(L):
            for jj in range(ii + 1, L):
                if rgb[ii, jj, 0] == 1.0 and rgb[ii, jj, 1] == 1.0 and rgb[ii, jj, 2] == 1.0:
                    rgb[ii, jj] = [0.92, 0.92, 0.92]
        ax3.imshow(rgb, aspect='equal')
        ax3.set_xlim(-0.5, L - 0.5)
        ax3.set_ylim(L - 0.5, -0.5)
        f1 = 2 * len(TP_pairs) / (2 * len(TP_pairs) + len(FP_pairs) + len(FN_pairs)) if (TP_pairs or FP_pairs or FN_pairs) else 0
        if m_idx == 0:
            ax3.set_title('Final Predictions\n(TP / FP / FN)', fontsize=14, fontweight='bold', pad=14)
        ax3.tick_params(labelsize=9)
        legend_elements = [
            Patch(facecolor=[0.2, 0.8, 0.2], label=f'TP ({len(TP_pairs)})'),
            Patch(facecolor=[0.9, 0.2, 0.2], label=f'FP ({len(FP_pairs)})'),
            Patch(facecolor=[0.2, 0.2, 0.9], label=f'FN ({len(FN_pairs)})'),
        ]
        leg = ax3.legend(handles=legend_elements, loc='lower left', fontsize=9,
                         framealpha=0.9, edgecolor='gray', handlelength=1,
                         bbox_to_anchor=(0.0, 0.0))
        ax3.text(0.02, 0.18, f'F1 = {f1:.2f}', transform=ax3.transAxes,
                 ha='left', va='bottom', fontsize=11, fontweight='bold')

    fig.subplots_adjust(hspace=0.35, wspace=0.30)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    fname = f'probe_contact_heatmaps{suffix}'
    for ext in ['png', 'pdf']:
        out_path = out_dir / f'{fname}.{ext}'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved {out_path}")
    plt.close()
    print(f"\nSequence: {seq_id}")
    return 0


def select_sequences(min_len=50, max_len=100, n=3):
    """Select TS0 sequences with length in range and reasonable base pairs."""
    bpRNA = {}
    with open(DATA / 'bpRNA.csv') as f:
        for r in csv.DictReader(f):
            bpRNA[r['id']] = {'seq': r['sequence'], 'pairs': r['base_pairs']}
    splits = {}
    with open(DATA / 'bpRNA_splits.csv') as f:
        for r in csv.DictReader(f):
            if r.get('partition', '').strip() == 'TS0':
                splits[r['id']] = True
    candidates = []
    for sid in splits:
        if sid not in bpRNA:
            continue
        L = len(bpRNA[sid]['seq'])
        if min_len <= L <= max_len:
            pairs = ast.literal_eval(bpRNA[sid]['pairs'])
            candidates.append((sid, L, len(pairs)))
    candidates.sort(key=lambda x: -x[2])
    return [c[0] for c in candidates[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(MARCH1 / 'figures' / 'probe_heatmaps'),
                    help='Output directory')
    ap.add_argument('--n-seq', type=int, default=3, help='Number of sequences (2-3)')
    ap.add_argument('--min-len', type=int, default=50)
    ap.add_argument('--max-len', type=int, default=100)
    ap.add_argument('--seq-ids', nargs='+', default=None, help='Override: specific seq IDs')
    ap.add_argument('--overlay', action='store_true', help='Add overlay panel (GT + pred@τ) instead of TP/FP/FN')
    ap.add_argument('--no-tp-fp-fn', action='store_true', help='Skip TP/FP/FN panel (only GT + Pred)')
    ap.add_argument('--seq-figure', type=str, default=None,
                    help='Seq ID only: create folder probe_heatmaps_{seq_lower} and save one figure (models=rows, 4 cols)')
    ap.add_argument('--models', nargs='+', default=None,
                    help='Filter models (e.g. --models ernie roberta onehot). Used with --seq-figure.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_cfg = load_best_config()
    seq_ids = args.seq_ids or select_sequences(args.min_len, args.max_len, args.n_seq)

    bpRNA = {}
    with open(DATA / 'bpRNA.csv') as f:
        for r in csv.DictReader(f):
            bpRNA[r['id']] = {'seq': r['sequence'], 'pairs': r['base_pairs']}

    # --seq-figure mode
    if args.seq_figure:
        seq_id = args.seq_figure
        folder_name = f'probe_heatmaps_{seq_id.lower().replace("-", "_")}'
        seq_out_dir = MARCH1 / 'figures' / folder_name
        seq_out_dir.mkdir(parents=True, exist_ok=True)
        model_filter = args.models
        suffix = '_selected' if model_filter else ''
        return _run_seq_figure(seq_id, seq_out_dir, best_cfg, bpRNA,
                               model_filter=model_filter, suffix=suffix)

    n_models = len(MODELS)
    n_seqs = len(seq_ids)
    # Columns: GT | Pred | Pred>=tau | TP/FP/FN (if not --no-tp-fp-fn)
    n_cols = 3 if args.no_tp_fp_fn else 4
    n_rows = n_seqs * n_models

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    # Per-model figures: each model gets its own figure (n_seqs rows × n_cols)
    per_model = {}
    for model in MODELS:
        if model not in best_cfg:
            continue
        fm, am = plt.subplots(n_seqs, n_cols, figsize=(5 * n_cols, 4 * n_seqs), squeeze=False)
        per_model[model] = (fm, am)

    for row_idx, seq_id in enumerate(seq_ids):
        if seq_id not in bpRNA:
            print(f"[WARN] {seq_id} not in bpRNA")
            continue
        seq = bpRNA[seq_id]['seq']
        pairs = ast.literal_eval(bpRNA[seq_id]['pairs'])
        L = len(seq)
        M_gt = pairs_to_contact_map(pairs, L)

        print(f"\n{seq_id} | L={L} | {len(pairs)} base pairs")

        for m_idx, model in enumerate(MODELS):
            if model not in best_cfg:
                continue
            cfg = best_cfg[model]
            layer, k, tau = cfg['layer'], cfg['k'], cfg['threshold']
            mode = cfg.get('decoding_mode', 'unconstrained')

            emb_path = get_embedding_path(seq_id, model, layer)
            ckpt_path = get_probe_path(model, layer, k)
            if not emb_path.exists():
                print(f"  [WARN] Missing embedding: {emb_path}")
                continue
            if not ckpt_path.exists():
                print(f"  [WARN] Missing checkpoint: {ckpt_path}")
                continue

            emb = np.load(emb_path)
            B, _, _ = load_probe_matrix(str(ckpt_path))
            B = B.cpu()
            P = compute_P(emb, B)  # returns numpy
            P_torch = torch.from_numpy(P).float()

            # Predicted pairs at τ (for TP/FP/FN)
            if mode == 'unconstrained':
                allowed = None
            else:
                allowed = create_canonical_mask(seq, allow_gu=(mode == 'canonical_wobble'))
            pred_pairs = prob_to_pairs(P_torch, threshold=tau, allowed_mask=allowed, inputs_are_logits=False)
            TP_pairs, FP_pairs, FN_pairs = compute_tp_fp_fn(pred_pairs, pairs)

            # Upper triangle only for display (i < j)
            P_ut = np.triu(P, 1) + np.tril(np.nan * np.ones_like(P))
            M_gt_ut = np.triu(M_gt, 1) + np.tril(np.nan * np.ones_like(M_gt))

            ax_row = row_idx * n_models + m_idx  # row in axes grid

            def _plot_panels(ax_list, show_model_in_title=True):
                """ax_list: [GT, Pred, Pred≥τ, TP/FP/FN]. TP/FP/FN can be None if --no-tp-fp-fn."""
                ax0, ax1, ax2, ax3 = ax_list[0], ax_list[1], ax_list[2], ax_list[3] if len(ax_list) > 3 else None

                ax0.imshow(M_gt_ut, cmap='Blues', aspect='equal', vmin=0, vmax=1)
                ax0.set_xlabel('Nucleotide position')
                ax0.set_ylabel('Nucleotide position')
                ax0.set_title(f'{MODEL_LABELS[model]} — Ground truth' if show_model_in_title else 'Ground truth')
                ax0.set_xlim(-0.5, L - 0.5)
                ax0.set_ylim(L - 0.5, -0.5)

                # Pred: full range [0,1]
                im1 = ax1.imshow(P_ut, cmap='hot', aspect='equal', vmin=0, vmax=1)
                ax1.set_xlabel('Nucleotide position')
                ax1.set_ylabel('Nucleotide position')
                ax1.set_title(f'{MODEL_LABELS[model]} — Predicted (τ={tau})' if show_model_in_title else f'Predicted (τ={tau})')
                ax1.set_xlim(-0.5, L - 0.5)
                ax1.set_ylim(L - 0.5, -0.5)
                plt.colorbar(im1, ax=ax1, label='p_ij')

                # Use vmin=tau to darken sub-threshold values.
                im2 = ax2.imshow(P_ut, cmap='hot', aspect='equal', vmin=tau, vmax=1)
                ax2.set_xlabel('Nucleotide position')
                ax2.set_ylabel('Nucleotide position')
                ax2.set_title(f'{MODEL_LABELS[model]} — Pred ≥ τ' if show_model_in_title else 'Pred ≥ τ')
                ax2.set_xlim(-0.5, L - 0.5)
                ax2.set_ylim(L - 0.5, -0.5)
                plt.colorbar(im2, ax=ax2, label='p_ij')

                if ax3 is not None:
                    rgb = np.ones((L, L, 3)) * 0.92
                    for (i, j) in TP_pairs:
                        if 0 <= i - 1 < L and 0 <= j - 1 < L and i < j:
                            rgb[i - 1, j - 1] = [0.2, 0.8, 0.2]
                    for (i, j) in FP_pairs:
                        if 0 <= i - 1 < L and 0 <= j - 1 < L and i < j:
                            rgb[i - 1, j - 1] = [0.9, 0.2, 0.2]
                    for (i, j) in FN_pairs:
                        if 0 <= i - 1 < L and 0 <= j - 1 < L and i < j:
                            rgb[i - 1, j - 1] = [0.2, 0.2, 0.9]
                    lower_mask = np.broadcast_to(np.tril(np.ones((L, L), dtype=bool))[:, :, np.newaxis], (L, L, 3))
                    rgb_masked = np.ma.masked_where(lower_mask, rgb)
                    ax3.imshow(rgb_masked, aspect='equal')
                    ax3.set_xlabel('Nucleotide position')
                    ax3.set_ylabel('Nucleotide position')
                    f1 = 2 * len(TP_pairs) / (2 * len(TP_pairs) + len(FP_pairs) + len(FN_pairs)) if (TP_pairs or FP_pairs or FN_pairs) else 0
                    ax3.set_title(f'{MODEL_LABELS[model]} — TP/FP/FN (F1={f1:.2f})' if show_model_in_title else f'TP/FP/FN (F1={f1:.2f})')
                    ax3.set_xlim(-0.5, L - 0.5)
                    ax3.set_ylim(L - 0.5, -0.5)
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor=[0.2, 0.8, 0.2], label=f'TP ({len(TP_pairs)})'),
                        Patch(facecolor=[0.9, 0.2, 0.2], label=f'FP ({len(FP_pairs)})'),
                        Patch(facecolor=[0.2, 0.2, 0.9], label=f'FN ({len(FN_pairs)})'),
                    ]
                    ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)

            ax_list = [axes[ax_row, 0], axes[ax_row, 1], axes[ax_row, 2]]
            if n_cols >= 4:
                ax_list.append(axes[ax_row, 3])
            _plot_panels(ax_list, show_model_in_title=True)

            if model in per_model:
                am = per_model[model][1]
                am_list = [am[row_idx, 0], am[row_idx, 1], am[row_idx, 2]]
                if n_cols >= 4:
                    am_list.append(am[row_idx, 3])
                _plot_panels(am_list, show_model_in_title=False)
                am[row_idx, 0].set_ylabel(f'{seq_id} L={L}', fontsize=9)

    # Row labels: seq_id | model
    for ax_row in range(n_rows):
        seq_idx = ax_row // n_models
        m_idx = ax_row % n_models
        seq_id = seq_ids[seq_idx]
        model = MODELS[m_idx]
        L = len(bpRNA.get(seq_id, {}).get('seq', '')) if seq_id in bpRNA else 0
        axes[ax_row, 0].set_ylabel(f'{seq_id} L={L}\n{MODEL_LABELS[model]}', fontsize=9)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        out_path = out_dir / f'probe_contact_heatmaps.{ext}'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved {out_path}")

    plt.close()

    # Save per-model figures
    for model, (fm, am) in per_model.items():
        fm.suptitle(MODEL_LABELS[model], fontsize=12)
        fm.tight_layout()
        for ext in ['png', 'pdf']:
            out_path = out_dir / f'probe_contact_heatmaps_{model}.{ext}'
            fm.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved {out_path}")
        plt.close(fm)
    print(f"\nSequences: {seq_ids}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
