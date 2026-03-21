# .pt Checkpoint and Probe Matrix Structure

## 1. .pt Checkpoint Structure

The `.pt` file is a PyTorch checkpoint (torch.save/torch.load). It typically contains:

```python
ckpt = {
    'model_state_dict': { ... },  # or 'state_dict', or direct keys
    # Possible keys: 'proj.weight', 'module.proj.weight', etc.
}
```

The loader in `jan22/scripts/generation/generate_base_pairs.py` tries multiple structures:
- `ckpt['model_state_dict']`
- `ckpt['state_dict']`
- Direct dict with keys containing `'proj'` and `'weight'`

**Key used**: `proj.weight` (or `module.proj.weight` etc.)

## 2. Probe Matrix B

**Shape**: `(k, d)` where
- `k` = projection dimension (e.g., 32, 64, 128)
- `d` = embedding dimension (model-dependent)

**Computation**:
1. `z_i = B @ h_i`  →  z_i ∈ ℝ^k
2. `s_ij = z_i^T z_j`  (dot product)
3. `p_ij = sigmoid(s_ij)`  →  pair probability

So the bilinear probe computes: `p_ij = σ(B h_i · B h_j)`.

## 3. Pseudoknot in Probe-Only

**No explicit pseudoknot constraint.**

Probe-only uses **greedy max-1 decoding** (`prob_to_pairs` in jan22/utils/evaluation.py):
- Select highest-probability pair above threshold
- Remove that row and column (1-to-1 constraint: each base pairs with at most one partner)
- Repeat until no valid pairs remain

This enforces **1-to-1** (each base at most one pair) but does **not** forbid crossing pairs (pseudoknots). The greedy algorithm can produce pseudoknots in principle.

**Canonical constraint** (`allowed_mask`): restricts pairs to AU, CG, (GU) — not pseudoknot-related.
