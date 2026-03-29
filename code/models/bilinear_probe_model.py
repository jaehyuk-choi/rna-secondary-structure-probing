"""Bilinear and low-rank pairwise probes for contact prediction."""
import torch
import torch.nn as nn


class BilinearContactProbe(nn.Module):
    """Full bilinear score for each pair: h_i^T W h_j. Logits for BCEWithLogitsLoss; encoder stays frozen elsewhere."""

    def __init__(self, input_dim: int):
        super().__init__()

        self.W = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)

    def forward(self, batch_embeddings: torch.Tensor, mask: torch.Tensor | None = None):
        transformed = torch.matmul(batch_embeddings, self.W)
        logits = torch.matmul(transformed, batch_embeddings.transpose(1, 2))

        if mask is not None:
            valid = mask.unsqueeze(1) & mask.unsqueeze(2)
            logits = logits.masked_fill(~valid, -1e9)

        return logits


class StructuralContactProbe(nn.Module):
    """Low-rank probe: project D-dim tokens to k dims, then pairwise dots give logits (k << D)."""

    def __init__(self, input_dim: int, proj_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim, bias=False)

    def forward(self, batch_embeddings: torch.Tensor, mask: torch.Tensor | None = None):
        z = self.proj(batch_embeddings)
        logits = torch.matmul(z, z.transpose(1, 2))

        if mask is not None:
            valid = mask.unsqueeze(1) & mask.unsqueeze(2)
            logits = logits.masked_fill(~valid, -1e9)

        return logits
