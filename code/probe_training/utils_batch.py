"""Pad sequences to batch tensors; misc split helpers."""
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd


def collate_rna_batch(embeddings_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [emb.shape[0] for emb in embeddings_list]
    padded = pad_sequence(embeddings_list, batch_first=True)

    max_len = padded.shape[1]
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    mask = torch.arange(max_len)[None, :] < length_tensor[:, None]

    return padded, mask


def build_lofo_folds(
    df: pd.DataFrame,
    id_col: str = "id",
    seed: int = 42,
):
    df = df.copy()
    df["family"] = df[id_col].astype(str).apply(lambda x: x.split("_")[0])

    print("\n=== Family distribution (pseudo-family from ID) ===")
    family_counts = df["family"].value_counts().rename("count")
    family_percent = (family_counts / len(df) * 100).rename("percent")
    summary = pd.concat([family_counts, family_percent], axis=1)
    summary = summary.sort_values("count", ascending=False)
    print(summary)
    print("Total sequences:", len(df))
    print("===================================================\n")

    big_fams = ["5s", "srp", "tRNA", "tmRNA", "RNaseP"]
    df_big = df[df["family"].isin(big_fams)].reset_index(drop=True)

    print("Keeping only big families for LOFO:", big_fams)
    print("Counts within big families:")
    print(df_big["family"].value_counts(), "\n")

    folds = []
    rng = random.Random(seed)
    for fam in big_fams:
        test_ids = df_big[df_big["family"] == fam][id_col].tolist()
        trainval_ids = df_big[df_big["family"] != fam][id_col].tolist()
        rng.shuffle(trainval_ids)
        split = int(0.9 * len(trainval_ids))
        train_ids = sorted(trainval_ids[:split])
        val_ids = sorted(trainval_ids[split:])
        folds.append((train_ids, val_ids, test_ids, fam))

    return folds


def build_train_val_test_split(
    df: pd.DataFrame,
    id_col: str = "id",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    all_ids = df[id_col].tolist()
    rng = random.Random(seed)
    rng.shuffle(all_ids)

    n_total = len(all_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_ids = sorted(all_ids[:n_train])
    val_ids = sorted(all_ids[n_train:n_train + n_val])
    test_ids = sorted(all_ids[n_train + n_val:])

    print(f"\n=== Random Split (8:1:1) ===")
    print(f"Total sequences: {n_total}")
    print(f"Train: {len(train_ids)} ({len(train_ids)/n_total*100:.1f}%)")
    print(f"Val: {len(val_ids)} ({len(val_ids)/n_total*100:.1f}%)")
    print(f"Test: {len(test_ids)} ({len(test_ids)/n_total*100:.1f}%)")
    print("=" * 30)

    return train_ids, val_ids, test_ids


def upper_triangle_mask(n: int) -> np.ndarray:
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def extract_upper_triangle(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    L = tensor.shape[0]
    triu = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tensor.device), diagonal=1)
    if mask is not None:
        triu = triu & mask
    return tensor[triu]
