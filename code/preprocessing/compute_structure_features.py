"""
Compute structure features from base pairs.
"""
import ast
import os

import numpy as np
import pandas as pd

# Base directories for bpRNA
ROOT = "/projects/u6cg/jay/dissertations"
DATA_PATH = os.path.join(ROOT, "data", "bpRNA.csv")
OUT_DIR = os.path.join(ROOT, "data", "contact_maps", "bpRNA")
os.makedirs(OUT_DIR, exist_ok=True)


def compute_contact_matrix(n, base_pairs):
    """
    Compute binary contact map for base-paired nucleotides.

    Args:
        n (int): RNA length.
        base_pairs (list of tuples): Base pairs (1-based index).

    Returns:
        np.ndarray: [n, n] binary matrix (1 for paired, 0 otherwise)
    """
    d_G = np.zeros((n, n), dtype=np.float32)
    for (i, j) in base_pairs:
        i, j = i - 1, j - 1
        d_G[i, j] = 1.0
        d_G[j, i] = 1.0
    return d_G


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} sequences from {DATA_PATH}")

    for idx, row in df.iterrows():
        rid = row["id"]
        base_pairs = ast.literal_eval(row["base_pairs"])
        # bpRNA.csv does not ship with an explicit length column, so compute it.
        n = int(row.get("len", len(row["sequence"])))

        contact_map = compute_contact_matrix(n, base_pairs)
        out_path = os.path.join(OUT_DIR, f"{rid}_contact.npy")
        np.save(out_path, contact_map)
        print(f"[{idx+1}/{len(df)}] Saved {out_path}")


if __name__ == "__main__":
    main()
