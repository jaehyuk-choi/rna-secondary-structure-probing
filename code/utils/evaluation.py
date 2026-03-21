# evaluation.py
from typing import Tuple, Optional, List
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available. Using matplotlib for confusion matrix visualization.")


def is_canonical_pair(base_i: str, base_j: str, allow_gu: bool = False) -> bool:
    """
    Check if two bases form a canonical base pair.
    
    Args:
        base_i: First base (A, U, G, C)
        base_j: Second base (A, U, G, C)
        allow_gu: If True, also allow GU/UG pairs (wobble pairs)
    
    Returns:
        True if the pair is canonical
    """
    base_i = base_i.upper()
    base_j = base_j.upper()
    
    # Standard canonical pairs: AU, UA, GC, CG
    canonical_pairs = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')}
    
    if allow_gu:
        # Add wobble pairs: GU, UG
        canonical_pairs.update({('G', 'U'), ('U', 'G')})
    
    return (base_i, base_j) in canonical_pairs


def create_canonical_mask(sequence: str, allow_gu: bool = False, device: str = "cpu") -> torch.Tensor:
    """
    Create a boolean mask [L, L] where True indicates canonical pairing is allowed.
    
    Args:
        sequence: RNA sequence string (0-indexed, will be converted to 1-indexed for pairs)
        allow_gu: If True, also allow GU/UG pairs
        device: Device for the tensor
    
    Returns:
        [L, L] boolean tensor where allowed_mask[i, j] = True if (i+1, j+1) is canonical
        Only upper triangle (i < j) is set to True for canonical pairs, rest is False
    """
    L = len(sequence)
    allowed_mask = torch.zeros(L, L, dtype=torch.bool, device=device)
    
    # Only check upper triangle (i < j)
    for i in range(L):
        for j in range(i + 1, L):
            base_i = sequence[i]
            base_j = sequence[j]
            if is_canonical_pair(base_i, base_j, allow_gu=allow_gu):
                allowed_mask[i, j] = True
    
    return allowed_mask


def compute_canonical_rate(pred_pairs: List[Tuple[int, int]], sequence: str, allow_gu: bool = False) -> float:
    """
    Compute the fraction of predicted pairs that are canonical.
    
    Args:
        pred_pairs: List of (i, j) pairs (1-based indices)
        sequence: RNA sequence string (0-indexed)
        allow_gu: If True, also allow GU/UG pairs
    
    Returns:
        CanonicalRate = (# canonical pairs) / (total predicted pairs)
        Returns NaN if no pairs predicted
    """
    if len(pred_pairs) == 0:
        return float('nan')
    
    canonical_count = 0
    for i, j in pred_pairs:
        # Convert 1-based to 0-based
        base_i = sequence[i - 1]
        base_j = sequence[j - 1]
        if is_canonical_pair(base_i, base_j, allow_gu=allow_gu):
            canonical_count += 1
    
    return canonical_count / len(pred_pairs)


def compute_pair_metrics(
    values: torch.Tensor,
    contact_map: torch.Tensor,
    threshold: float = 0.0,
    shift: int = 1,
    sequence: Optional[str] = None,
    allow_gu: bool = False,
    allowed_mask: Optional[torch.Tensor] = None,
    inputs_are_logits: bool = False,
):
    """
    Compute TP, FP, FN allowing shift tolerance.
    Accepts (i±shift, j) or (i, j±shift) as a correct pair.

    Equivalent to dot_metrics(correct_pair(..., shift)),
    but using contact-map tensors instead of dot-bracket.
    
    Args:
        values: [L, L] logits/probabilities tensor
        contact_map: [L, L] contact map tensor
        threshold: Threshold for pair prediction (ALWAYS in probability space, 0-1)
        shift: Shift tolerance for matching
        sequence: Optional RNA sequence string for canonical rate calculation
        allow_gu: If True, allow GU/UG pairs in canonical rate calculation
        allowed_mask: Optional [L, L] boolean mask for canonical constraint (for onehot)
        inputs_are_logits: If True, values are logits (will be converted to probabilities).
                          If False, values are already probabilities (0-1).
    
    Returns:
        Tuple of (TP, FP, FN) or (TP, FP, FN, canonical_rate) if sequence is provided
    """
    pred_pairs = prob_to_pairs(values, threshold=threshold, allowed_mask=allowed_mask, inputs_are_logits=inputs_are_logits)
    true_pairs = contact_to_pairs(contact_map)

    TP = FP = FN = 0

    # --- match predicted pairs ---
    for (pi, pj) in pred_pairs:
        matched = False
        for (ti, tj) in true_pairs:
            if (abs(pi - ti) <= shift and pj == tj) or \
               (pi == ti and abs(pj - tj) <= shift):
                matched = True
                break
        
        if matched:
            TP += 1
        else:
            FP += 1

    # --- find FN (true but none predicted close enough) ---
    for (ti, tj) in true_pairs:
        matched = False
        for (pi, pj) in pred_pairs:
            if (abs(pi - ti) <= shift and pj == tj) or \
               (pi == ti and abs(pj - tj) <= shift):
                matched = True
                break
        
        if not matched:
            FN += 1

    # Compute canonical rate if sequence is provided
    if sequence is not None:
        canonical_rate = compute_canonical_rate(pred_pairs, sequence, allow_gu=allow_gu)
        return TP, FP, FN, canonical_rate
    
    return TP, FP, FN


# def compute_pair_metrics(
#     values: torch.Tensor,
#     contact_map: torch.Tensor,
#     threshold: float = 0.0,
# ) -> Tuple[int, int, int]:
#     """
#     Compute TP, FP, FN from a SINGLE sequence using a positive-centric view.

#     This version:
#       * converts logits into a set of predicted positive pairs (i, j)
#       * converts the gold contact map into a set of true positive pairs (i, j)
#       * computes TP, FP, FN from these two sets

#     It does NOT iterate over every negative cell in the full L x L matrix.
#     """
#     # 1) Convert to pair lists (1-based, i < j only)
#     pred_pairs = prob_to_pairs(values, threshold=threshold)
#     true_pairs = contact_to_pairs(contact_map) # only the true positive pairs

#     # 2) Use sets for strict matching (no shift tolerance)
#     pred_set = set(pred_pairs)
#     true_set = set(true_pairs)

#     # True positives: predicted and gold
#     TP = len(pred_set & true_set)
#     # False positives: predicted but not in gold
#     FP = len(pred_set - true_set)
#     # False negatives: gold but not predicted
#     FN = len(true_set - pred_set)

#     return TP, FP, FN



def precision_recall_f1(TP: int, FP: int, FN: int):
    """
    Compute precision, recall, F1 from TP, FP, FN.
    """
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def prob_to_pairs(
    logits: torch.Tensor,
    threshold: float = 0.0,
    allowed_mask: Optional[torch.Tensor] = None,
    inputs_are_logits: bool = False,
):
    """
    Convert an [L, L] logits matrix into a list of predicted base pairs.
    Uses greedy global argmax to enforce global 1-to-1 constraint:
    - Selects the highest probability pair from the entire matrix
    - Removes the corresponding row and column to prevent conflicts
    - Repeats until no valid pairs remain above threshold
    
    This ensures both row and column constraints are satisfied simultaneously,
    guaranteeing that each base pairs with at most one partner.

    Args:
        logits: [L, L] logits/probabilities tensor
        threshold: Threshold for pair prediction (ALWAYS in probability space, 0-1)
        allowed_mask: Optional [L, L] boolean mask. If provided, only pairs where
                     allowed_mask[i, j] == True can be selected. Used for canonical
                     constraint in onehot decoding.
        inputs_are_logits: If True, applies sigmoid to convert logits to probabilities.
                          If False, assumes inputs are already probabilities (0-1).
    
    Returned indices are 1-based and only include i < j.
    """
    L = logits.shape[0]
    device = logits.device

    # Convert logits to probabilities if needed
    if inputs_are_logits:
        # Apply sigmoid to convert logits to probabilities
        values = torch.sigmoid(logits)
    else:
        # Already probabilities, use as-is
        values = logits

    # Use only the upper triangle (i < j)
    triu_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1)

    # Create a working copy of probabilities with lower triangle masked
    working_logits = values.clone()
    working_logits[~triu_mask] = float('-inf')
    
    # Apply canonical constraint mask if provided (for onehot canonical_constrained mode)
    if allowed_mask is not None:
        # Mask out non-canonical pairs by setting them to -inf
        working_logits[~allowed_mask] = float('-inf')

    # Track which rows and columns are still available
    available_rows = torch.ones(L, dtype=torch.bool, device=device)
    available_cols = torch.ones(L, dtype=torch.bool, device=device)

    pred_pairs = []

    # Greedy selection: repeatedly find the global maximum
    while True:
        # Create mask for available cells (both row and column must be available)
        available_mask = available_rows.unsqueeze(1) & available_cols.unsqueeze(0)
        # Combine with upper triangle constraint
        valid_mask = available_mask & triu_mask
        
        # Apply canonical constraint if provided
        if allowed_mask is not None:
            valid_mask = valid_mask & allowed_mask
        
        # Apply mask to working logits (temporarily mask unavailable cells)
        masked_logits = working_logits.clone()
        masked_logits[~valid_mask] = float('-inf')
        
        # Find global maximum
        flat_idx = torch.argmax(masked_logits.view(-1))
        i = flat_idx // L
        j = flat_idx % L
        max_prob = masked_logits[i, j]
        
        # Check if we found a valid pair above threshold
        if max_prob <= threshold or max_prob == float('-inf'):
            break
        
        # Add the pair (convert to 1-based)
        pred_pairs.append((int(i) + 1, int(j) + 1))
        
        # Remove row i and column j from further consideration
        # by masking them in the working logits
        working_logits[i, :] = float('-inf')
        working_logits[:, j] = float('-inf')
        available_rows[i] = False
        available_cols[j] = False

    return pred_pairs


def contact_to_pairs(contact_map: torch.Tensor):
    """
    Convert a gold contact map [L, L] (0/1) into a list of true base pairs.

    Returned indices are 1-based and only include i < j.
    """
    L = contact_map.shape[0]
    device = contact_map.device

    # Use only the upper triangle (i < j)
    triu_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1)

    # True positive locations (gold base pairs)
    pos_mask = (contact_map > 0) & triu_mask

    idx = pos_mask.nonzero(as_tuple=False)
    true_pairs = [(int(i) + 1, int(j) + 1) for i, j in idx]
    return true_pairs


def plot_confusion_matrix(y_true, y_pred, save_path, family_name, threshold):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        save_path: Path to save the figure
        family_name: Name of the family for title
        threshold: Threshold used for predictions
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
    else:
        # Fallback to matplotlib imshow
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Negative', 'Positive'])
        plt.yticks(tick_marks, ['Negative', 'Positive'])
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {family_name} (threshold={threshold})')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve(y_true, y_prob, save_path, family_name, threshold):
    """
    Plot and save Precision-Recall curve.
    
    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        save_path: Path to save the figure
        family_name: Name of the family for title
        threshold: Threshold used for predictions
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - {family_name} (threshold={threshold})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path, family_name, threshold):
    """
    Plot and save ROC curve.
    
    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        save_path: Path to save the figure
        family_name: Name of the family for title
        threshold: Threshold used for predictions
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {family_name} (threshold={threshold})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_sigmoid_hist(scores, save_path, family_name, threshold):
    """
    Plot and save sigmoid output distribution histogram.
    
    Args:
        scores: Sigmoid output scores
        save_path: Path to save the figure
        family_name: Name of the family for title
        threshold: Threshold used for predictions
    """
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=50, edgecolor='black')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
    plt.xlabel('Sigmoid Output')
    plt.ylabel('Frequency')
    plt.title(f'Sigmoid Output Distribution - {family_name} (threshold={threshold})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()