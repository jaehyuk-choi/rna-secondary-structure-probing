#!/usr/bin/env python3
"""Roll up Val α=0 vs best-α and TS0/NEW metrics at Val-optimal α."""
import csv
from collections import defaultdict
from pathlib import Path

models = ['ernie', 'roberta', 'rnafm', 'rinalmo', 'onehot', 'rnabert']

def agg_at_alpha(csv_path, target_alpha, tol=0.005):
    d = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            a = float(row['alpha'])
            if abs(a - target_alpha) <= tol:
                d[a]['tp'] += int(row['tp'])
                d[a]['fp'] += int(row['fp'])
                d[a]['fn'] += int(row['fn'])
    if not d:
        return None
    best_a = min(d.keys(), key=lambda x: abs(x - target_alpha))
    t, fp, fn = d[best_a]['tp'], d[best_a]['fp'], d[best_a]['fn']
    rec = t / (t + fn) if (t + fn) > 0 else 0
    prec = t / (t + fp) if (t + fp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return (prec, rec, f1)

def get_best_alpha_and_a0(csv_dir):
    out_best = {}
    out_a0 = {}
    for m in models:
        p = Path(csv_dir) / f'detailed_results_{m}.csv'
        if not p.exists() or p.stat().st_size == 0:
            continue
        d = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        with open(p) as f:
            r = csv.DictReader(f)
            for row in r:
                a = float(row['alpha'])
                d[a]['tp'] += int(row['tp'])
                d[a]['fp'] += int(row['fp'])
                d[a]['fn'] += int(row['fn'])
        best_a, best_f1 = None, -1
        a0_f1 = None
        for a in sorted(d.keys()):
            t, fp, fn = d[a]['tp'], d[a]['fp'], d[a]['fn']
            rec = t / (t + fn) if (t + fn) > 0 else 0
            prec = t / (t + fp) if (t + fp) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1, best_a = f1, a
            if abs(a) < 0.01:
                a0_f1 = f1
        out_best[m] = (best_a, best_f1)
        out_a0[m] = a0_f1
    return out_best, out_a0

def main():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    folding = REPO_ROOT / 'results' / 'folding'

    # VL0 validation sweep results
    val_v = folding / 'results_vl0'
    val_c = folding / 'results_vl0_contrafold'
    best_v, a0_v = get_best_alpha_and_a0(val_v)
    best_c, a0_c = get_best_alpha_and_a0(val_c)

    # TS0, NEW held-out evaluation
    ts0_v = folding / 'results_ts0'
    ts0_c = folding / 'results_ts0_contrafold'
    new_v = folding / 'results_new'
    new_c = folding / 'results_new_contrafold'

    models_ts0_new = [m for m in models if m != 'rnabert']  # TS0/NEW excludes rnabert

    print("=" * 80)
    print("CPLfold Val: Alpha=0 vs Optimal Alpha")
    print("=" * 80)
    print()
    print("| Model | Backend | Alpha=0 F1 | Best alpha | Optimal F1 | Delta |")
    print("|-------|---------|------------|------------|------------|-------|")
    for m in models:
        if m in best_v:
            a0 = a0_v.get(m)
            ba, bf = best_v[m]
            delta = (bf - a0) if a0 is not None else None
            a0s = f"{a0:.4f}" if a0 is not None else "N/A"
            ds = f"{delta:+.4f}" if delta is not None else "N/A"
            print(f"| {m} | Vienna | {a0s} | {ba:.2f} | {bf:.4f} | {ds} |")
        if m in best_c:
            a0 = a0_c.get(m)
            ba, bf = best_c[m]
            delta = (bf - a0) if a0 is not None else None
            a0s = f"{a0:.4f}" if a0 is not None else "N/A"
            ds = f"{delta:+.4f}" if delta is not None else "N/A"
            print(f"| {m} | Contrafold | {a0s} | {ba:.2f} | {bf:.4f} | {ds} |")

    print()
    print("=" * 80)
    print("TS0 / NEW: Results at the validation-selected optimal alpha")
    print("(Val optimal alpha from VL0; TS0/NEW held-out evaluation)")
    print("=" * 80)
    print()
    print("| Model | Backend | Val best α | TS0 F1 | NEW F1 |")
    print("|-------|---------|------------|--------|--------|")

    for m in models_ts0_new:
        if m in best_v:
            ba = best_v[m][0]
            ts0 = agg_at_alpha(ts0_v / f'detailed_results_{m}.csv', ba)
            new_r = agg_at_alpha(new_v / f'detailed_results_{m}.csv', ba)
            ts0s = f"{ts0[2]:.4f}" if ts0 else "N/A"
            news = f"{new_r[2]:.4f}" if new_r else "N/A"
            print(f"| {m} | Vienna | {ba:.2f} | {ts0s} | {news} |")
        if m in best_c:
            ba = best_c[m][0]
            ts0 = agg_at_alpha(ts0_c / f'detailed_results_{m}.csv', ba)
            new_r = agg_at_alpha(new_c / f'detailed_results_{m}.csv', ba)
            ts0s = f"{ts0[2]:.4f}" if ts0 else "N/A"
            news = f"{new_r[2]:.4f}" if new_r else "N/A"
            print(f"| {m} | Contrafold | {ba:.2f} | {ts0s} | {news} |")

    print()
    print("=" * 80)
    print("TS0 / NEW: Alpha=0 (no bonus, baseline)")
    print("=" * 80)
    print()
    print("| Model | Backend | TS0 F1 (α=0) | NEW F1 (α=0) |")
    print("|-------|---------|--------------|--------------|")
    for m in models_ts0_new:
        if m in best_v:
            ts0 = agg_at_alpha(ts0_v / f'detailed_results_{m}.csv', 0.0)
            new_r = agg_at_alpha(new_v / f'detailed_results_{m}.csv', 0.0)
            ts0s = f"{ts0[2]:.4f}" if ts0 else "N/A"
            news = f"{new_r[2]:.4f}" if new_r else "N/A"
            print(f"| {m} | Vienna | {ts0s} | {news} |")
        if m in best_c:
            ts0 = agg_at_alpha(ts0_c / f'detailed_results_{m}.csv', 0.0)
            new_r = agg_at_alpha(new_c / f'detailed_results_{m}.csv', 0.0)
            ts0s = f"{ts0[2]:.4f}" if ts0 else "N/A"
            news = f"{new_r[2]:.4f}" if new_r else "N/A"
            print(f"| {m} | Contrafold | {ts0s} | {news} |")

if __name__ == '__main__':
    main()
