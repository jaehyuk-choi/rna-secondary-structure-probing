#!/usr/bin/env python3
"""Box plot of bpRNA sequence lengths by partition (TR0, VL0, TS0, NEW)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]

splits = {}
with open(REPO_ROOT / 'data' / 'splits' / 'bpRNA_splits.csv') as f:
    r = csv.DictReader(f)
    for row in r:
        p = row.get('partition', '').strip().upper()
        if p:
            splits[row['id']] = p

bpRNA = {}
with open(REPO_ROOT / 'data' / 'metadata' / 'bpRNA.csv') as f:
    r = csv.DictReader(f)
    for row in r:
        seq = row.get('sequence', row.get('seq', ''))
        bpRNA[row['id']] = len(seq)

by_part = {'TR0': [], 'VL0': [], 'TS0': [], 'NEW': []}
for sid, part in splits.items():
    if sid in bpRNA and part in by_part:
        by_part[part].append(bpRNA[sid])

fig, ax = plt.subplots(figsize=(8, 5))
parts = ['TR0', 'VL0', 'TS0', 'NEW']
data = [by_part[p] for p in parts]
bp = ax.boxplot(data, tick_labels=parts, patch_artist=True, showfliers=True)

colors = ['#4A90D9', '#81B29A', '#E07A5F', '#9B8EC2']
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)

ax.set_ylabel('Sequence length (nt)', fontsize=12)
ax.set_xlabel('Partition', fontsize=12)
ax.set_title('bpRNA Sequence Length by Partition', fontsize=14)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

out_dir = REPO_ROOT / 'figures' / 'main'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'bpRNA_length_boxplot.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
