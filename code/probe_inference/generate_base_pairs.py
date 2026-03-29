#!/usr/bin/env python3
"""Write base_pair.txt for CPLfold from a probe checkpoint and per-seq embeddings."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def load_probe_matrix(checkpoint_path, device='cpu'):
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    state_dict = None
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif any('proj' in str(k) for k in ckpt.keys()):
            state_dict = ckpt
        else:
            state_dict = ckpt
    
    if state_dict is None:
        raise ValueError(f"Could not find state dict in checkpoint. Available keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'Not a dict'}")
    
    proj_key = None
    for key in state_dict.keys():
        if 'proj' in key.lower() and 'weight' in key.lower():
            proj_key = key
            break
    
    if proj_key is None:
        raise ValueError(f"Could not find projection weight in checkpoint. Available keys: {list(state_dict.keys())}")
    
    B = state_dict[proj_key]
    print(f"Found projection matrix at key: {proj_key}")
    print(f"Projection matrix shape: {B.shape}")
    
    if len(B.shape) != 2:
        raise ValueError(f"Expected 2D projection matrix, got shape {B.shape}")
    
    k, d = B.shape
    print(f"Projection dimension k={k}, Embedding dimension d={d}")
    
    return B, k, d


def load_embeddings(embedding_path, device='cpu'):
    print(f"Loading embeddings from: {embedding_path}")
    emb = np.load(embedding_path)
    
    if len(emb.shape) != 2:
        raise ValueError(f"Expected 2D embedding array (L, d), got shape {emb.shape}")
    
    L, d_emb = emb.shape
    print(f"Embedding shape: ({L}, {d_emb})")
    
    emb_tensor = torch.from_numpy(emb).float().to(device)
    
    return emb_tensor, L, d_emb


def compute_pair_scores(embeddings, B, device='cpu'):
    print(f"Computing pair scores...")
    B = B.to(device)
    z = torch.matmul(embeddings, B.t())
    s_ij = torch.matmul(z, z.t())
    p_ij = torch.sigmoid(s_ij)
    
    print(f"Computed pair score matrix of shape {p_ij.shape}")
    print(f"Score range: [{p_ij.min().item():.4f}, {p_ij.max().item():.4f}]")
    
    return p_ij


def filter_pairs(p_ij, threshold=None, topk=None):
    L = p_ij.shape[0]
    pairs = []
    for i in range(L):
        for j in range(i + 1, L):
            score = p_ij[i, j].item()
            pairs.append((i + 1, j + 1, score))
    
    print(f"Total pairs (i < j): {len(pairs)}")
    
    if topk is not None:
        pairs.sort(key=lambda x: x[2], reverse=True)
        pairs = pairs[:topk]
        print(f"After top-{topk} filter: {len(pairs)} pairs")
    
    if threshold is not None:
        pairs = [(i, j, s) for i, j, s in pairs if s >= threshold]
        print(f"After threshold >= {threshold} filter: {len(pairs)} pairs")
    
    return pairs


def write_base_pair_file(pairs, output_path):
    print(f"Writing {len(pairs)} pairs to: {output_path}")
    
    with open(output_path, 'w') as f:
        for i, j, score in pairs:
            f.write(f"{i}\t{j}\t{score:.6f}\n")
    
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate base_pair.txt from probe checkpoint and nucleotide embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        print("warn: CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
    
    # Validate input files
    if not Path(args.checkpoint).exists():
        print(f"error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    if not Path(args.embeddings).exists():
        print(f"error: Embedding file not found: {args.embeddings}")
        sys.exit(1)
    
    try:
        # Load probe matrix
        B, k, d = load_probe_matrix(args.checkpoint, device=args.device)
        
        # Load embeddings
        embeddings, L, d_emb = load_embeddings(args.embeddings, device=args.device)
        
        # Validate dimension match
        if d != d_emb:
            print(f"error: Dimension mismatch: checkpoint expects d={d}, but embeddings have d={d_emb}")
            sys.exit(1)
        
        # Compute pair scores
        p_ij = compute_pair_scores(embeddings, B, device=args.device)
        
        # Filter pairs
        pairs = filter_pairs(p_ij, threshold=args.threshold, topk=args.topk)
        
        if len(pairs) == 0:
            print("warn: no pairs after filter (try --threshold / --topk)")
        
        # Write output
        write_base_pair_file(pairs, args.output)
        
        print(f"wrote {args.output}")
        
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

