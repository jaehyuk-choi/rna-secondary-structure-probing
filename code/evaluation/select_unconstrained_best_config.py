#!/usr/bin/env python3
"""Pick best (layer,k,τ) from val unconstrained sweeps; writes config CSV and runs probe-only metrics."""

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / 'results' / 'outputs'
CONFIG_DIR = REPO_ROOT / 'configs'
METRICS_DIR = REPO_ROOT / 'results' / 'metrics'


def find_unconstrained_val_sweep(run_dir: Path) -> Path | None:
    for name in ['val_threshold_sweep_unconstrained.csv', 'val_threshold_sweep.csv']:
        p = run_dir / name
        if p.exists():
            return p
    return None


def get_best_unconstrained_per_run(run_dir: Path, model: str) -> dict | None:
    path = find_unconstrained_val_sweep(run_dir)
    if not path:
        return None
    best = None
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get('decoding_mode', '').strip().lower() != 'unconstrained':
                continue
            f1 = float(row.get('f1', -1))
            if best is None or f1 > best['val_f1']:
                best = {
                    'model': model,
                    'layer': int(row.get('layer', 0)),
                    'k': int(row.get('k', 0)),
                    'seed': int(row.get('seed', 42)),
                    'threshold': float(row.get('threshold', 0)),
                    'decoding_mode': 'unconstrained',
                    'val_f1': f1,
                    'val_precision': float(row.get('precision', 0)),
                    'val_recall': float(row.get('recall', 0)),
                }
    return best


def select_unconstrained_best_config():
    """Select best config per model (unconstrained only)."""
    models = ['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert']
    best_per_model = {}

    for model in models:
        model_dir = OUTPUTS / model
        if not model_dir.exists():
            print(f"warn: No outputs for {model}")
            continue
        global_best = None
        for run_dir in model_dir.rglob('seed_42'):
            if not (run_dir / 'best.pt').exists():
                continue
            res = get_best_unconstrained_per_run(run_dir, model)
            if res and (global_best is None or res['val_f1'] > global_best['val_f1']):
                global_best = res
        if global_best:
            best_per_model[model] = global_best
            print(f"{model}: L{global_best['layer']} k{global_best['k']} thresh={global_best['threshold']:.2f} Val F1={global_best['val_f1']:.4f}")

    return best_per_model


def write_config_csv(best_per_model: dict, out_path: Path):
    """Write config CSV for compute_probe_only_metrics."""
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'model', 'selected_layer', 'selected_k', 'selected_seed',
            'selected_best_threshold', 'selection_rule', 'selected_decoding_mode'
        ])
        w.writeheader()
        for m, cfg in best_per_model.items():
            w.writerow({
                'model': m,
                'selected_layer': cfg['layer'],
                'selected_k': cfg['k'],
                'selected_seed': cfg['seed'],
                'selected_best_threshold': cfg['threshold'],
                'selection_rule': 'max_val_f1_unconstrained',
                'selected_decoding_mode': 'unconstrained',
            })


def run_probe_only_metrics(config_csv: Path, output_dir: Path):
    """Run compute_probe_only_metrics with custom config."""
    script = REPO_ROOT / 'code' / 'evaluation' / 'compute_probe_only_metrics.py'
    cmd = [
        sys.executable, str(script),
        '--config-csv', str(config_csv),
        '--checkpoint-base', str(OUTPUTS),
        '--output-dir', str(output_dir),
        '--progress-log', str(output_dir / 'probe_only_progress.log'),
    ]
    subprocess.run(cmd, check=True)


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config_csv = CONFIG_DIR / 'final_selected_config_unconstrained.csv'
    output_dir = METRICS_DIR

    print("=" * 60)
    print("best unconstrained config per model")
    print("=" * 60)
    best_per_model = select_unconstrained_best_config()
    if not best_per_model:
        print("error: No configs found")
        return 1

    write_config_csv(best_per_model, config_csv)
    print(f"\nWrote config: {config_csv}")

    print("\n" + "=" * 60)
    print("compute_probe_only_metrics (TS0/NEW)")
    print("=" * 60)
    run_probe_only_metrics(config_csv, output_dir)

    # Load TS0/NEW results and build summary table
    test_path = output_dir / 'final_test_metrics.csv'
    new_path = output_dir / 'final_new_metrics.csv'
    if not test_path.exists() or not new_path.exists():
        print("error: TS0/NEW results not found")
        return 1

    ts0 = {}
    with open(test_path) as f:
        for row in csv.DictReader(f):
            ts0[row['model']] = row
    new = {}
    with open(new_path) as f:
        for row in csv.DictReader(f):
            new[row['model']] = row

    # Write summary table
    summary_path = METRICS_DIR / 'unconstrained_results_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        fieldnames = [
            'model', 'layer', 'k', 'threshold', 'decoding_mode',
            'val_f1', 'val_precision', 'val_recall',
            'ts0_f1', 'ts0_precision', 'ts0_recall',
            'new_f1', 'new_precision', 'new_recall',
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in sorted(best_per_model.keys()):
            cfg = best_per_model[m]
            t = ts0.get(m, {})
            n = new.get(m, {})
            w.writerow({
                'model': m,
                'layer': cfg['layer'],
                'k': cfg['k'],
                'threshold': f"{cfg['threshold']:.2f}",
                'decoding_mode': 'unconstrained',
                'val_f1': f"{cfg['val_f1']:.4f}",
                'val_precision': f"{cfg['val_precision']:.4f}",
                'val_recall': f"{cfg['val_recall']:.4f}",
                'ts0_f1': t.get('f1', ''),
                'ts0_precision': t.get('precision', ''),
                'ts0_recall': t.get('recall', ''),
                'new_f1': n.get('f1', ''),
                'new_precision': n.get('precision', ''),
                'new_recall': n.get('recall', ''),
            })
    print(f"\nWrote summary: {summary_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
