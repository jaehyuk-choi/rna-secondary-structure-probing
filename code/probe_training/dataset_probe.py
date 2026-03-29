"""Dataset: one seq → embedding [L,D] + contact [L,L] from npy on disk."""
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils_batch import collate_rna_batch


class RNABasepairDataset(Dataset):

    def __init__(
        self,
        ids: List[str],
        embedding_dir: str,
        contact_dir: str,
        embedding_suffix: str = ".npy",
        contact_suffix: str = "_contact.npy",
    ):
        self.ids = ids
        self.embedding_dir = embedding_dir
        self.contact_dir = contact_dir
        self.embedding_suffix = embedding_suffix
        self.contact_suffix = contact_suffix

    def __len__(self) -> int:
        return len(self.ids)

    def _load_embedding(self, rid: str) -> torch.Tensor:
        path = os.path.join(self.embedding_dir, f"{rid}{self.embedding_suffix}")
        arr = np.load(path)  # [L, D]
        return torch.from_numpy(arr).float()

    def _load_contact(self, rid: str) -> torch.Tensor:
        path = os.path.join(self.contact_dir, f"{rid}{self.contact_suffix}")
        arr = np.load(path)  # [L, L]
        return torch.from_numpy(arr).float()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        rid = self.ids[idx]
        emb = self._load_embedding(rid)
        contact = self._load_contact(rid)
        return emb, contact, rid


def rna_collate_fn(batch):
    """Custom collate for DataLoader.

    Returns ``(padded_emb, contacts_list, mask, ids)`` where
    *contacts_list* is kept as a Python list of variable-size tensors.
    """
    embs, contacts, ids = zip(*batch)
    padded_embs, mask = collate_rna_batch(list(embs))
    return padded_embs, list(contacts), mask, list(ids)
