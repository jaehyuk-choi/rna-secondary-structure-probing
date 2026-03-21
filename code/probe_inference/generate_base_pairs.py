#!/usr/bin/env python3
"""
Inference script to generate base_pair.txt files from probe checkpoints and nucleotide embeddings.

This script:
1. Loads a probe checkpoint (.pt) and extracts the projection matrix B
2. Loads nucleotide embeddings (.npy) for a sequence
3. Computes pair scores: z_i = B h_i, s_ij = z_i^T z_j, p_ij = sigmoid(s_ij)
4. Writes base_pair.txt in the format required by CPLfold_inter (1-based indices, i < j)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def load_probe_matrix(checkpoint_path, device='cpu'):
    """
    Load probe checkpoint and extract the projection matrix B.
    
    The checkpoint may have different structures:
    - {'model_state_dict': {'proj.weight': ...}}
    - {'proj.weight': ...}
    - Direct state dict with 'proj.weight'
    
    Returns:
        B (torch.Tensor): Projection matrix of shape (k, d)
        k (int): Projection dimension
        d (int): Embedding dimension
    """
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Try different possible key structures
    state_dict = None
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif any('proj' in str(k) for k in ckpt.keys()):
            state_dict = ckpt
        else:
            # Try treating the whole dict as state_dict
            state_dict = ckpt
    
    if state_dict is None:
        raise ValueError(f"Could not find state dict in checkpoint. Available keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'Not a dict'}")
    
    # Find the projection weight key (could be 'proj.weight', 'module.proj.weight', etc.)
    proj_key = None
    for key in state_dict.keys():
        if 'proj' in key.lower() and 'weight' in key.lower():
            proj_key = key
            break
    
    if proj_key is None:
        raise ValueError(f"Could not find projection weight in checkpoint. Available keys: {list(state_dict.keys())}")
    
    B = state_dict[proj_key]
    print(f"[INFO] Found projection matrix at key: {proj_key}")
    print(f"[INFO] Projection matrix shape: {B.shape}")
    
    # B should be (k, d) where k is projection dim and d is embedding dim
    if len(B.shape) != 2:
        raise ValueError(f"Expected 2D projection matrix, got shape {B.shape}")
    
    k, d = B.shape
    print(f"[INFO] Projection dimension k={k}, Embedding dimension d={d}")
    
    return B, k, d


def load_embeddings(embedding_path, device='cpu'):
    """
    Load nucleotide embeddings from .npy file.
    
    Args:
        embedding_path: Path to .npy file
        device: Device to load on
        
    Returns:
        embeddings (torch.Tensor): Shape (L, d) where L is sequence length
        L (int): Sequence length
    """
    print(f"[INFO] Loading embeddings from: {embedding_path}")
    emb = np.load(embedding_path)
    
    if len(emb.shape) != 2:
        raise ValueError(f"Expected 2D embedding array (L, d), got shape {emb.shape}")
    
    L, d_emb = emb.shape
    print(f"[INFO] Embedding shape: ({L}, {d_emb})")
    
    # Convert to torch tensor
    emb_tensor = torch.from_numpy(emb).float().to(device)
    
    return emb_tensor, L, d_emb


def compute_pair_scores(embeddings, B, device='cpu'):
    """
    Compute pair scores using the probe formulation:
    z_i = B h_i
    s_ij = z_i^T z_j
    p_ij = sigmoid(s_ij)
    
    Args:
        embeddings: (L, d) tensor of nucleotide embeddings
        B: (k, d) projection matrix
        device: Device to compute on
        
    Returns:
        p_ij: (L, L) tensor of pair probabilities (symmetric)
    """
    print(f"[INFO] Computing pair scores...")
    
    # Move B to device if needed
    B = B.to(device)
    
    # z_i = B h_i: (L, d) @ (k, d)^T -> (L, k)
    # Note: B is (k, d), so we need B @ h_i^T, which is (k, d) @ (d, L) -> (k, L)
    # Actually, we want z_i for each position: (L, d) @ (d, k) = (L, k)
    # So we need B^T: (L, d) @ (d, k) = (L, k)
    z = torch.matmul(embeddings, B.t())  # (L, k)
    
    # s_ij = z_i^T z_j: (L, k) @ (k, L) -> (L, L)
    s_ij = torch.matmul(z, z.t())  # (L, L)
    
    # p_ij = sigmoid(s_ij)
    p_ij = torch.sigmoid(s_ij)
    
    print(f"[INFO] Computed pair score matrix of shape {p_ij.shape}")
    print(f"[INFO] Score range: [{p_ij.min().item():.4f}, {p_ij.max().item():.4f}]")
    
    return p_ij


def filter_pairs(p_ij, threshold=None, topk=None):
    """
    Filter pairs to keep only i < j and apply optional filtering.
    
    Args:
        p_ij: (L, L) tensor of pair probabilities
        threshold: Optional probability threshold (keep pairs with p >= threshold)
        topk: Optional top-K filter (keep top-K highest scoring pairs)
        
    Returns:
        pairs: List of tuples (i, j, score) with 1-based indices where i < j
    """
    L = p_ij.shape[0]
    
    # Extract upper triangle (i < j) with 1-based indices
    pairs = []
    for i in range(L):
        for j in range(i + 1, L):
            score = p_ij[i, j].item()
            pairs.append((i + 1, j + 1, score))  # 1-based indices
    
    print(f"[INFO] Total pairs (i < j): {len(pairs)}")
    
    # Apply topk filter first (if specified)
    if topk is not None:
        pairs.sort(key=lambda x: x[2], reverse=True)
        pairs = pairs[:topk]
        print(f"[INFO] After top-{topk} filter: {len(pairs)} pairs")
    
    # Apply threshold filter (if specified)
    if threshold is not None:
        pairs = [(i, j, s) for i, j, s in pairs if s >= threshold]
        print(f"[INFO] After threshold >= {threshold} filter: {len(pairs)} pairs")
    
    return pairs


def write_base_pair_file(pairs, output_path):
    """
    Write base_pair.txt file in the format required by CPLfold_inter.
    
    Format:
    i    j    score
    (tab-separated, 1-based indices)
    
    Args:
        pairs: List of tuples (i, j, score) with 1-based indices
        output_path: Path to output file
    """
    print(f"[INFO] Writing {len(pairs)} pairs to: {output_path}")
    
    with open(output_path, 'w') as f:
        for i, j, score in pairs:
            f.write(f"{i}\t{j}\t{score:.6f}\n")
    
    print(f"[INFO] Successfully wrote base_pair.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Generate base_pair.txt from probe checkpoint and nucleotide embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python generate_base_pairs.py checkpoint.pt embeddings.npy -o base_pair.txt
  
  # With threshold filtering
  python generate_base_pairs.py checkpoint.pt embeddings.npy -o base_pair.txt --threshold 0.5
  
  # With top-K filtering
  python generate_base_pairs.py checkpoint.pt embeddings.npy -o base_pair.txt --topk 100
  
  # Using CUDA
  python generate_base_pairs.py checkpoint.pt embeddings.npy -o base_pair.txt --device cuda
  
  # Then run CPLfold_inter:
  python CPLfold_inter.py <SEQ> --bonus base_pair.txt --alpha 0 --V    # baseline
  python CPLfold_inter.py <SEQ> --bonus base_pair.txt --alpha 0.5 --V  # with probe support
  python CPLfold_inter.py <SEQ> --bonus base_pair.txt --alpha 1.0 --V
  python CPLfold_inter.py <SEQ> --bonus base_pair.txt --alpha 2.0 --V
        """
    )
    
    parser.add_argument('checkpoint', type=str, help='Path to probe checkpoint (.pt file)')
    parser.add_argument('embeddings', type=str, help='Path to nucleotide embeddings (.npy file)')
    parser.add_argument('-o', '--output', type=str, default='base_pair.txt',
                       help='Output path for base_pair.txt (default: base_pair.txt)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Probability threshold (keep pairs with p >= threshold)')
    parser.add_argument('--topk', type=int, default=None,
                       help='Keep only top-K highest scoring pairs')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for computation (default: cpu)')
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
    
    # Validate input files
    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    if not Path(args.embeddings).exists():
        print(f"[ERROR] Embedding file not found: {args.embeddings}")
        sys.exit(1)
    
    try:
        # Load probe matrix
        B, k, d = load_probe_matrix(args.checkpoint, device=args.device)
        
        # Load embeddings
        embeddings, L, d_emb = load_embeddings(args.embeddings, device=args.device)
        
        # Validate dimension match
        if d != d_emb:
            print(f"[ERROR] Dimension mismatch: checkpoint expects d={d}, but embeddings have d={d_emb}")
            sys.exit(1)
        
        # Compute pair scores
        p_ij = compute_pair_scores(embeddings, B, device=args.device)
        
        # Filter pairs
        pairs = filter_pairs(p_ij, threshold=args.threshold, topk=args.topk)
        
        if len(pairs) == 0:
            print("[WARNING] No pairs remaining after filtering. Consider adjusting --threshold or --topk")
        
        # Write output
        write_base_pair_file(pairs, args.output)
        
        print(f"[INFO] Done! Generated {args.output}")
        
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

