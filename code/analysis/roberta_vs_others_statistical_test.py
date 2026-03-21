#!/usr/bin/env python3
"""
Statistical comparison: RoBERTa vs other models (ERNIE, RNAFM, RiNALMo, One-hot, RNABERT).
Paired tests by sequence. Uses best config (filters by final_selected_config).
"""

import csv
from pathlib import Path

import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

MARCH1 = Path(__file__).resolve().parents[1]
OUT_DIR = MARCH1 / 'figures' / 'roberta_significance'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEB8 = MARCH1.parent / 'feb8/results_updated/summary'
MARCH1_DATA = MARCH1 / 'data'
BEST_CONFIG_PATH = FEB8 / 'final_selected_config.csv'

MODELS = ['roberta', 'ernie', 'rnafm', 'rinalmo', 'onehot', 'rnabert']
MODEL_LABELS = {'roberta': 'RoBERTa', 'ernie': 'ERNIE', 'rnafm': 'RNAFM', 'rinalmo': 'RiNALMo',
                'onehot': 'One-hot', 'rnabert': 'RNABERT'}


def _per_seq_paths():
    ts = MARCH1_DATA / 'ts_per_sequence_metrics.csv'
    new = MARCH1_DATA / 'new_per_sequence_metrics.csv'
    return (ts if ts.exists() else FEB8 / 'ts_per_sequence_metrics.csv',
            new if new.exists() else FEB8 / 'new_per_sequence_metrics.csv')


def load_best_config():
    cfg = {}
    with open(BEST_CONFIG_PATH) as f:
        for r in csv.DictReader(f):
            m = r['model']
            if m == 'model':
                continue
            cfg[m] = {
                'layer': int(r['selected_layer']),
                'k': int(r['selected_k']),
                'threshold': float(r['selected_best_threshold']),
                'decoding_mode': r['selected_decoding_mode'],
            }
    return cfg


def load_pivot(path, best_cfg):
    """Load per_sequence, filter to best config, pivot to seq_id -> {model: f1}."""
    by_seq = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m = r.get('model', '')
            if m not in MODELS or m not in best_cfg:
                continue
            bc = best_cfg[m]
            if (int(r.get('layer', -1)) != bc['layer'] or
                    int(r.get('k', -1)) != bc['k'] or
                    abs(float(r.get('threshold', -1)) - bc['threshold']) > 1e-6 or
                    r.get('decoding_mode', '') != bc['decoding_mode']):
                continue
            sid = r['seq_id']
            f1 = float(r.get('f1', 0) or 0)
            if sid not in by_seq:
                by_seq[sid] = {}
            by_seq[sid][m] = f1
    return by_seq


def paired_test(roberta_vals, other_vals):
    roberta_vals = np.array(roberta_vals)
    other_vals = np.array(other_vals)
    mean_diff = np.mean(roberta_vals - other_vals)
    if not HAS_SCIPY:
        return mean_diff, float('nan'), float('nan')
    try:
        _, p_wilcoxon = stats.wilcoxon(roberta_vals, other_vals, alternative='greater')
    except Exception:
        p_wilcoxon = float('nan')
    try:
        _, p_ttest = stats.ttest_rel(roberta_vals, other_vals, alternative='greater')
    except Exception:
        p_ttest = float('nan')
    return mean_diff, p_wilcoxon, p_ttest


def sig_str(p):
    if np.isnan(p) or p > 0.05:
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def main():
    ts_path, new_path = _per_seq_paths()
    if not ts_path.exists() or not new_path.exists():
        print("Per-sequence metrics not found. Run:")
        print("  python feb8/scripts/evaluation/compute_feb8_probe_only_metrics.py --per-sequence")
        return 1
    best_cfg = load_best_config()
    ts_data = load_pivot(ts_path, best_cfg)
    new_data = load_pivot(new_path, best_cfg)
    n_ts_roberta = sum(1 for v in ts_data.values() if 'roberta' in v)
    if n_ts_roberta < 10:
        print("WARNING: per_sequence does not have best config for RoBERTa.")
        print("Run: python feb8/scripts/evaluation/compute_feb8_probe_only_metrics.py --per-sequence")
        return 1

    other_models = [m for m in MODELS if m != 'roberta']  # RoBERTa vs all others including ERNIE
    rows = []
    for partition, data in [('TS0 (TEST)', ts_data), ('NEW', new_data)]:
        for other in other_models:
            roberta_vals = []
            other_vals = []
            for sid, model_f1 in data.items():
                if 'roberta' in model_f1 and other in model_f1:
                    roberta_vals.append(model_f1['roberta'])
                    other_vals.append(model_f1[other])
            if len(roberta_vals) < 10:
                rows.append({'partition': partition, 'comparison': f'RoBERTa vs {MODEL_LABELS[other]}',
                            'n': len(roberta_vals), 'mean_roberta': np.nan, 'mean_other': np.nan,
                            'mean_diff': np.nan, 'p_wilcoxon': np.nan, 'p_ttest': np.nan, 'sig': ''})
                continue
            mean_diff, p_wilcoxon, p_ttest = paired_test(roberta_vals, other_vals)
            rows.append({'partition': partition, 'comparison': f'RoBERTa vs {MODEL_LABELS[other]}',
                        'n': len(roberta_vals), 'mean_roberta': np.mean(roberta_vals),
                        'mean_other': np.mean(other_vals), 'mean_diff': mean_diff,
                        'p_wilcoxon': p_wilcoxon, 'p_ttest': p_ttest, 'sig': sig_str(p_wilcoxon)})

    csv_path = OUT_DIR / 'roberta_vs_others_significance.csv'
    fieldnames = ['partition', 'comparison', 'n', 'mean_roberta', 'mean_other', 'mean_diff', 'p_wilcoxon', 'p_ttest', 'sig']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r[k] for k in fieldnames}
            for k in ['mean_roberta', 'mean_other', 'mean_diff', 'p_wilcoxon', 'p_ttest']:
                if isinstance(out[k], float) and not np.isnan(out[k]):
                    out[k] = f'{out[k]:.4f}'
            w.writerow(out)
    print(f"Saved {csv_path}")

    md_path = OUT_DIR / 'roberta_vs_others_significance.md'
    with open(md_path, 'w') as f:
        f.write("# RoBERTa vs Others: Paired Statistical Test (Sequence-level F1)\n\n")
        f.write("| Partition | Comparison | N | Mean RoBERTa | Mean Other | Δ | p (Wilcoxon) | Sig |\n")
        f.write("|-----------|------------|---|--------------|------------|---|---------------|-----|\n")
        for r in rows:
            mr = f"{r['mean_roberta']:.4f}" if not np.isnan(r['mean_roberta']) else "-"
            mo = f"{r['mean_other']:.4f}" if not np.isnan(r['mean_other']) else "-"
            md = f"{r['mean_diff']:.4f}" if not np.isnan(r['mean_diff']) else "-"
            pw = f"{r['p_wilcoxon']:.2e}" if not np.isnan(r['p_wilcoxon']) else "-"
            f.write(f"| {r['partition']} | {r['comparison']} | {r['n']} | {mr} | {mo} | {md} | {pw} | {r['sig']} |\n")
    print(f"Saved {md_path}")
    print("Done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
