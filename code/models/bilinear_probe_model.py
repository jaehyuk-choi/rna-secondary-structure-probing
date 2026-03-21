# bilinear_probe_model.py
import torch
import torch.nn as nn


class BilinearContactProbe(nn.Module):
    """
    Bilinear structural probe for RNA base-pair contact prediction.
        s_ij = h_i^T W h_j
    This is then passed through a sigmoid (via BCEWithLogitsLoss) to obtain
    the probability that positions (i, j) form a base pair.

    Importantly:
    - RNABERT embeddings are NOT updated (frozen encoder).
    - Only W is trained, which makes this a probing model, not a full task model.
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim (int): Dimensionality of RNABERT embeddings (e.g., 768 or 1024).
        """
        super().__init__()

        # Learnable bilinear weight: W ∈ R^(D x D)
        # We keep it small scale at init for stability.
        self.W = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)

    def forward(self, batch_embeddings: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Compute pairwise logits for all (i, j) positions in each sequence.
        Args:
            batch_embeddings (Tensor): Shape [B, L, D]
                Batch of padded RNABERT embeddings.
            mask (Tensor, optional): Shape [B, L], dtype bool
                True for valid positions, False for padding.

        Returns:
            logits (Tensor): Shape [B, L, L]
                Raw scores (logits) for all pairwise contacts.
                You should pass these to BCEWithLogitsLoss.
        """
        # batch_embeddings: [B, L, D]
        B, L, D = batch_embeddings.shape

        # Apply bilinear form: s_ij = h_i^T W h_j
        # Step 1: (B, L, D) @ (D, D) -> (B, L, D)
        transformed = torch.matmul(batch_embeddings, self.W)

        # Step 2: (B, L, D) @ (B, D, L) -> (B, L, L)
        # torch.matmul supports batch matmul when the leading dims match.
        logits = torch.matmul(transformed, batch_embeddings.transpose(1, 2))

        if mask is not None:
            # mask: [B, L] -> pairwise mask: [B, L, L]
            valid = mask.unsqueeze(1) & mask.unsqueeze(2)  # broadcast
            # Optional: we usually do not want to score padding positions,
            # so we set them to a very negative value.
            logits = logits.masked_fill(~valid, -1e9)

        return logits

class StructuralContactProbe(nn.Module):
    """
    Low-rank structural probe for RNA base-pair contact prediction.

    h_i ∈ R^D  (RNABERT hidden state)
    B ∈ R^(D x k)  (k << D)

    z_i = h_i B   
    s_ij = z_i^T z_j

    In code:
      - Input: batch_embeddings [B, L, D]
      - proj: a Linear(D -> k) layer (no bias)
      - Output: logits [B, L, L]
    """

    def __init__(self, input_dim: int, proj_dim: int = 64):
        """
        input_dim: RNABERT hidden size D (e.g., 120)
        proj_dim: structural subspace size k (e.g., 16, 32, 64, etc.)
        """
        super().__init__()
        # B: R^(D x k), implemented as a linear projection (no bias needed)
        self.proj = nn.Linear(input_dim, proj_dim, bias=False)

    def forward(self, batch_embeddings: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            batch_embeddings: [B, L, D]
            mask: [B, L] (True = valid token, False = padding)

        Returns:
            logits: [B, L, L] (score for each token pair, before sigmoid)
        """
        # 1) Project into structural subspace: z_i = B h_i
        #    z: [B, L, k]
        z = self.proj(batch_embeddings)

        # 2) Pairwise dot products: [B, L, k] @ [B, k, L] -> [B, L, L]
        logits = torch.matmul(z, z.transpose(1, 2))

        # 3) If mask is provided, set invalid (padding) positions to a large negative value
        if mask is not None:
            valid = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, L, L]
            logits = logits.masked_fill(~valid, -1e9)

        return logits
