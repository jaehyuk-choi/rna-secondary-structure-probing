"""Metrics, pair extraction, and plots for probe/CPLfold evaluation."""
from typing import Optional, List

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
    """AU/GC/CG (+ optional GU/UG if allow_gu)."""
    base_i = base_i.upper()
    base_j = base_j.upper()

    canonical_pairs = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')}

    if allow_gu:
        canonical_pairs.update({('G', 'U'), ('U', 'G')})

    return (base_i, base_j) in canonical_pairs


def create_canonical_mask(sequence: str, allow_gu: bool = False, device: str = "cpu") -> torch.Tensor:
    """Upper-triangle mask: which (i,j) are allowed under WC or WC+GU rules."""
    L = len(sequence)
    allowed_mask = torch.zeros(L, L, dtype=torch.bool, device=device)

    for i in range(L):
        for j in range(i + 1, L):
            base_i = sequence[i]
            base_j = sequence[j]
            if is_canonical_pair(base_i, base_j, allow_gu=allow_gu):
                allowed_mask[i, j] = True

    return allowed_mask


def compute_canonical_rate(pred_pairs: List[Tuple[int, int]], sequence: str, allow_gu: bool = False) -> float:
    """Fraction of predicted pairs that pass is_canonical_pair. NaN if pred_pairs empty."""
    if len(pred_pairs) == 0:
        return float('nan')

    canonical_count = 0
    for i, j in pred_pairs:
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
    """TP/FP/FN with optional ±shift on one index; optional canonical_rate if sequence given."""
    pred_pairs = prob_to_pairs(values, threshold=threshold, allowed_mask=allowed_mask, inputs_are_logits=inputs_are_logits)
    true_pairs = contact_to_pairs(contact_map)

    TP = FP = FN = 0

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

    for (ti, tj) in true_pairs:
        matched = False
        for (pi, pj) in pred_pairs:
            if (abs(pi - ti) <= shift and pj == tj) or \
               (pi == ti and abs(pj - tj) <= shift):
                matched = True
                break

        if not matched:
            FN += 1

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
    """Standard P/R/F1 from counts."""
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
    """Greedy global argmax on upper triangle, respecting max-one partner per base; 1-based i<j."""
    L = logits.shape[0]
    device = logits.device

    if inputs_are_logits:
        values = torch.sigmoid(logits)
    else:
        values = logits

    triu_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1)

    working_logits = values.clone()
    working_logits[~triu_mask] = float('-inf')

    if allowed_mask is not None:
        working_logits[~allowed_mask] = float('-inf')

    available_rows = torch.ones(L, dtype=torch.bool, device=device)
    available_cols = torch.ones(L, dtype=torch.bool, device=device)

    pred_pairs = []

    while True:
        available_mask = available_rows.unsqueeze(1) & available_cols.unsqueeze(0)
        valid_mask = available_mask & triu_mask

        if allowed_mask is not None:
            valid_mask = valid_mask & allowed_mask

        masked_logits = working_logits.clone()
        masked_logits[~valid_mask] = float('-inf')

        flat_idx = torch.argmax(masked_logits.view(-1))
        i = flat_idx // L
        j = flat_idx % L
        max_prob = masked_logits[i, j]

        if max_prob <= threshold or max_prob == float('-inf'):
            break

        pred_pairs.append((int(i) + 1, int(j) + 1))

        working_logits[i, :] = float('-inf')
        working_logits[:, j] = float('-inf')
        available_rows[i] = False
        available_cols[j] = False

    return pred_pairs


def contact_to_pairs(contact_map: torch.Tensor):
    """Gold pairs from upper triangle of 0/1 map; 1-based i<j."""
    L = contact_map.shape[0]
    device = contact_map.device

    triu_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1)

    pos_mask = (contact_map > 0) & triu_mask

    idx = pos_mask.nonzero(as_tuple=False)
    true_pairs = [(int(i) + 1, int(j) + 1) for i, j in idx]
    return true_pairs


def plot_confusion_matrix(y_true, y_pred, save_path, family_name, threshold):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
    else:
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
