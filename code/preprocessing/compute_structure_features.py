"""Build binary contact npy files from bpRNA.csv base_pairs."""
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = str(REPO_ROOT / "data" / "metadata" / "bpRNA.csv")
OUT_DIR = str(REPO_ROOT / "data" / "contact_maps" / "bpRNA")
os.makedirs(OUT_DIR, exist_ok=True)


def compute_contact_matrix(n, base_pairs):
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
        n = int(row.get("len", len(row["sequence"])))

        contact_map = compute_contact_matrix(n, base_pairs)
        out_path = os.path.join(OUT_DIR, f"{rid}_contact.npy")
        np.save(out_path, contact_map)
        print(f"[{idx+1}/{len(df)}] Saved {out_path}")


if __name__ == "__main__":
    main()
