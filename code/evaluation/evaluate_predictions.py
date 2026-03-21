#!/usr/bin/env python3
"""
Evaluate predicted RNA structures against ground truth using jan22/utils/evaluation.py.

Computes F1, precision, recall for each model × alpha combination.
"""

import argparse
import ast
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Add jan22 utils to path
sys.path.insert(0, '/projects/u6cg/jay/dissertations/jan22')
from utils.evaluation import (
    compute_pair_metrics,
    contact_to_pairs,
    precision_recall_f1,
    prob_to_pairs,
)


def dot_bracket_to_pairs(structure, seq_length):
    """
    Convert dot-bracket notation to list of base pairs (1-based).
    
    Args:
        structure: Dot-bracket string (e.g., "(((...)))")
        seq_length: Sequence length
        
    Returns:
        List of tuples (i, j) where i < j (1-based indices)
    """
    pairs = []
    stack = []
    
    for i, char in enumerate(structure):
        pos = i + 1  # 1-based
        if char == '(':
            stack.append(pos)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((min(j, pos), max(j, pos)))
    
    return sorted(pairs)


def parse_base_pairs(base_pairs_str):
    """
    Parse base pairs from string format.
    
    Args:
        base_pairs_str: String like "[[2, 25], [4, 23], ...]"
        
    Returns:
        List of tuples (i, j) where i < j (1-based indices)
    """
    try:
        pairs_list = ast.literal_eval(base_pairs_str)
        # Convert to tuples and ensure i < j
        pairs = []
        for pair in pairs_list:
            if len(pair) >= 2:
                i, j = int(pair[0]), int(pair[1])
                pairs.append((min(i, j), max(i, j)))
        return sorted(pairs)
    except:
        return []


def pairs_to_contact_map(pairs, seq_length):
    """
    Convert list of base pairs to contact map tensor.
    
    Args:
        pairs: List of tuples (i, j) where i < j (1-based indices)
        seq_length: Sequence length
        
    Returns:
        [L, L] torch.Tensor with 1.0 at pair positions, 0.0 elsewhere
    """
    contact_map = torch.zeros(seq_length, seq_length, dtype=torch.float32)
    
    for i, j in pairs:
        # Convert 1-based to 0-based
        contact_map[i - 1, j - 1] = 1.0
        contact_map[j - 1, i - 1] = 1.0  # Symmetric
    
    return contact_map


def base_pair_file_to_prob_matrix(base_pair_file, seq_length):
    """
    Load base_pair.txt file and convert to probability matrix.
    
    Args:
        base_pair_file: Path to base_pair.txt file
        seq_length: Sequence length
        
    Returns:
        [L, L] torch.Tensor with probabilities
    """
    prob_matrix = torch.zeros(seq_length, seq_length, dtype=torch.float32)
    
    with open(base_pair_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                i = int(parts[0]) - 1  # Convert to 0-based
                j = int(parts[1]) - 1
                score = float(parts[2])
                if 0 <= i < seq_length and 0 <= j < seq_length:
                    prob_matrix[i, j] = score
                    prob_matrix[j, i] = score  # Symmetric
    
    return prob_matrix


def load_bpRNA_ground_truth(bpRNA_csv):
    """Load ground truth base pairs from bpRNA.csv."""
    ground_truth = {}
    with open(bpRNA_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row['id']
            base_pairs_str = row['base_pairs']
            pairs = parse_base_pairs(base_pairs_str)
            ground_truth[seq_id] = pairs
    return ground_truth


def save_to_detailed_csv(results, csv_file, seq_id, sequence, seq_length, ground_truth_pairs, threshold_used, use_threshold):
    """
    Save detailed evaluation results to CSV file (append mode).
    
    Args:
        results: List of result dictionaries from evaluation
        csv_file: Path to CSV file
        seq_id: Sequence ID
        sequence: RNA sequence
        seq_length: Sequence length
        ground_truth_pairs: Number of ground truth pairs
        threshold_used: Threshold value used (if any)
        use_threshold: Whether threshold was used
    """
    csv_file = Path(csv_file)
    file_exists = csv_file.exists()
    
    # Field names matching terminal output
    fieldnames = [
        'seq_id', 'sequence', 'seq_length', 'ground_truth_pairs',
        'model', 'alpha', 'threshold_used', 'use_threshold',
        'f1', 'precision', 'recall', 'tp', 'fp', 'fn', 
        'predicted_count', 'energy'
    ]
    
    # Open in append mode
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write each result
        for result in results:
            row = {
                'seq_id': seq_id,
                'sequence': sequence,
                'seq_length': seq_length,
                'ground_truth_pairs': ground_truth_pairs,
                'model': result['model'],
                'alpha': result['alpha'],
                'threshold_used': threshold_used if threshold_used is not None else '',
                'use_threshold': use_threshold,
                'f1': result['f1'],
                'precision': result['precision'],
                'recall': result['recall'],
                'tp': result['tp'],
                'fp': result['fp'],
                'fn': result['fn'],
                'predicted_count': result.get('predicted_count', ''),
                'energy': result.get('energy', '')
            }
            writer.writerow(row)
    
    if not file_exists:
        print(f"[INFO] Created detailed results CSV: {csv_file}")
    else:
        print(f"[INFO] Appended to detailed results CSV: {csv_file}")


def evaluate_summary(summary_file, bpRNA_csv, output_file=None, shift=1, use_base_pair_file=False):
    """
    Evaluate predictions from summary JSON file using jan22/utils/evaluation.py.
    
    Args:
        summary_file: Path to summary JSON file
        bpRNA_csv: Path to bpRNA.csv with ground truth
        output_file: Optional path to save evaluation results
        shift: Shift tolerance for pair matching (default: 1)
        use_base_pair_file: If True, use base_pair.txt probability matrix instead of dot-bracket
    """
    # Load summary
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    seq_id = summary['seq_id']
    sequence = summary['sequence']
    seq_length = len(sequence)
    
    # Load ground truth
    print(f"[INFO] Loading ground truth from: {bpRNA_csv}")
    ground_truth = load_bpRNA_ground_truth(bpRNA_csv)
    
    if seq_id not in ground_truth:
        print(f"[ERROR] Sequence ID not found in ground truth: {seq_id}")
        return None
    
    true_pairs = ground_truth[seq_id]
    print(f"[INFO] Ground truth has {len(true_pairs)} base pairs")
    
    # Convert ground truth to contact map
    true_contact_map = pairs_to_contact_map(true_pairs, seq_length)
    
    # Evaluate each model × alpha combination
    results = []
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {seq_id}")
    print(f"{'='*80}")
    print(f"Sequence length: {seq_length}")
    print(f"Ground truth pairs: {len(true_pairs)}")
    print(f"Shift tolerance: {shift}")
    print(f"\n{'Model':<12} {'Alpha':<8} {'F1':<8} {'Precision':<12} {'Recall':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-" * 80)
    
    for model_name, model_data in summary['models'].items():
        alpha_results = model_data.get('alpha_results', {})
        base_pair_file = model_data.get('base_pair_file')
        
        for alpha in summary['alphas']:
            # Alpha keys might be strings or floats
            alpha_key = str(alpha)
            if alpha_key not in alpha_results:
                if alpha not in alpha_results:
                    continue
                alpha_key = alpha
            
            result_data = alpha_results[alpha_key]
            structure = result_data.get('structure')
            
            if not structure:
                continue
            
            # Method 1: Use base_pair.txt probability matrix (if available and requested)
            predicted_pairs = None
            if use_base_pair_file and base_pair_file and Path(base_pair_file).exists():
                try:
                    prob_matrix = base_pair_file_to_prob_matrix(base_pair_file, seq_length)
                    
                    # Use evaluation.py's compute_pair_metrics with shift tolerance
                    TP, FP, FN = compute_pair_metrics(
                        prob_matrix,
                        true_contact_map,
                        threshold=0.0,  # Use all pairs (threshold already applied in base_pair.txt)
                        shift=shift,
                        inputs_are_logits=False  # Already probabilities
                    )
                    
                    precision, recall, f1 = precision_recall_f1(TP, FP, FN)
                    
                    # Get predicted pairs for count
                    predicted_pairs = prob_to_pairs(prob_matrix, threshold=0.0, inputs_are_logits=False)
                    
                except Exception as e:
                    print(f"[WARNING] Failed to use base_pair.txt for {model_name} alpha={alpha}: {e}")
                    # Fall back to dot-bracket method
                    predicted_pairs = dot_bracket_to_pairs(structure, seq_length)
                    pred_contact_map = pairs_to_contact_map(predicted_pairs, seq_length)
                    TP, FP, FN = compute_pair_metrics(
                        pred_contact_map,
                        true_contact_map,
                        threshold=0.0,
                        shift=shift,
                        inputs_are_logits=False
                    )
                    precision, recall, f1 = precision_recall_f1(TP, FP, FN)
            else:
                # Method 2: Use dot-bracket structure
                predicted_pairs = dot_bracket_to_pairs(structure, seq_length)
                pred_contact_map = pairs_to_contact_map(predicted_pairs, seq_length)
                
                TP, FP, FN = compute_pair_metrics(
                    pred_contact_map,
                    true_contact_map,
                    threshold=0.0,
                    shift=shift,
                    inputs_are_logits=False
                )
                precision, recall, f1 = precision_recall_f1(TP, FP, FN)
            
            results.append({
                'seq_id': seq_id,
                'model': model_name,
                'alpha': float(alpha),  # Ensure float
                'structure': structure,
                'energy': result_data.get('energy'),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'tp': int(TP),
                'fp': int(FP),
                'fn': int(FN),
                'predicted_count': len(predicted_pairs) if predicted_pairs else None
            })
            
            print(f"{model_name:<12} {alpha:<8.1f} {f1:<8.4f} {precision:<12.4f} {recall:<10.4f} {TP:<6} {FP:<6} {FN:<6}")
    
    print(f"{'='*80}")
    
    # Find best alpha for each model
    print(f"\n{'='*80}")
    print("BEST ALPHA PER MODEL (by F1)")
    print(f"{'='*80}")
    print(f"{'Model':<12} {'Best Alpha':<12} {'F1':<8} {'Precision':<12} {'Recall':<10}")
    print("-" * 60)
    
    model_best = {}
    for result in results:
        model = result['model']
        if model not in model_best or result['f1'] > model_best[model]['f1']:
            model_best[model] = result
    
    for model, best in sorted(model_best.items()):
        print(f"{model:<12} {best['alpha']:<12.1f} {best['f1']:<8.4f} {best['precision']:<12.4f} {best['recall']:<10.4f}")
    
    # Get threshold info from summary if available
    threshold_used = summary.get('threshold_used', None)
    use_threshold = summary.get('use_threshold', False)
    
    # Save results
    eval_results = {
        'seq_id': seq_id,
        'sequence': sequence,
        'seq_length': seq_length,
        'ground_truth_pairs': len(true_pairs),
        'shift_tolerance': shift,
        'threshold_used': threshold_used,
        'use_threshold': use_threshold,
        'results': results,
        'best_per_model': {k: {kk: vv for kk, vv in v.items() if kk != 'structure'} 
                          for k, v in model_best.items()}
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n[INFO] Evaluation results saved to: {output_file}")
    
    # Also save to detailed CSV (append mode for cumulative results)
    # Save to parent directory (not in out/ subdirectory) for easier access
    if output_file:
        output_path = Path(output_file)
        csv_file = output_path.parent.parent / 'detailed_results.csv'  # Go up from out/ to main output_dir
        save_to_detailed_csv(results, csv_file, seq_id, sequence, seq_length, len(true_pairs), threshold_used, use_threshold)
    
    return eval_results


def evaluate_all_summaries(summary_dir, bpRNA_csv, output_file=None, shift=1, use_base_pair_file=False):
    """Evaluate all summary files in a directory."""
    summary_dir = Path(summary_dir)
    summary_files = list(summary_dir.glob('summary_*.json'))
    
    print(f"[INFO] Found {len(summary_files)} summary files")
    
    all_results = []
    ground_truth = load_bpRNA_ground_truth(bpRNA_csv)
    
    for summary_file in summary_files:
        print(f"\n{'='*80}")
        print(f"Evaluating: {summary_file.name}")
        print(f"{'='*80}")
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        seq_id = summary['seq_id']
        sequence = summary['sequence']
        seq_length = len(sequence)
        
        if seq_id not in ground_truth:
            print(f"[WARNING] {seq_id} not found in ground truth, skipping")
            continue
        
        true_pairs = ground_truth[seq_id]
        true_contact_map = pairs_to_contact_map(true_pairs, seq_length)
        
        for model_name, model_data in summary['models'].items():
            alpha_results = model_data.get('alpha_results', {})
            base_pair_file = model_data.get('base_pair_file')
            
            for alpha in summary['alphas']:
                alpha_key = str(alpha)
                if alpha_key not in alpha_results:
                    if alpha not in alpha_results:
                        continue
                    alpha_key = alpha
                
                result_data = alpha_results[alpha_key]
                structure = result_data.get('structure')
                
                if not structure:
                    continue
                
                # Use evaluation.py methods
                predicted_pairs = dot_bracket_to_pairs(structure, seq_length)
                pred_contact_map = pairs_to_contact_map(predicted_pairs, seq_length)
                
                TP, FP, FN = compute_pair_metrics(
                    pred_contact_map,
                    true_contact_map,
                    threshold=0.0,
                    shift=shift,
                    inputs_are_logits=False
                )
                precision, recall, f1 = precision_recall_f1(TP, FP, FN)
                
                all_results.append({
                    'seq_id': seq_id,
                    'model': model_name,
                    'alpha': alpha,
                    'energy': result_data.get('energy'),
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': TP,
                    'fp': FP,
                    'fn': FN
                })
    
    # Aggregate results
    if all_results:
        print(f"\n{'='*80}")
        print("AGGREGATE RESULTS (across all sequences)")
        print(f"{'='*80}")
        
        # Average F1 per model × alpha
        model_alpha_f1 = defaultdict(list)
        for result in all_results:
            key = (result['model'], result['alpha'])
            model_alpha_f1[key].append(result['f1'])
        
        print(f"\n{'Model':<12} {'Alpha':<8} {'Avg F1':<10} {'Count':<8}")
        print("-" * 40)
        
        for (model, alpha), f1s in sorted(model_alpha_f1.items()):
            avg_f1 = sum(f1s) / len(f1s)
            print(f"{model:<12} {alpha:<8.1f} {avg_f1:<10.4f} {len(f1s):<8}")
        
        # Best alpha per model (average F1)
        print(f"\n{'='*80}")
        print("BEST ALPHA PER MODEL (average F1 across all sequences)")
        print(f"{'='*80}")
        print(f"{'Model':<12} {'Best Alpha':<12} {'Avg F1':<10}")
        print("-" * 40)
        
        model_best_alpha = {}
        for (model, alpha), f1s in model_alpha_f1.items():
            avg_f1 = sum(f1s) / len(f1s)
            if model not in model_best_alpha or avg_f1 > model_best_alpha[model]['avg_f1']:
                model_best_alpha[model] = {'alpha': alpha, 'avg_f1': avg_f1}
        
        for model, best in sorted(model_best_alpha.items()):
            print(f"{model:<12} {best['alpha']:<12.1f} {best['avg_f1']:<10.4f}")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[INFO] All evaluation results saved to: {output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate predicted RNA structures against ground truth using jan22/utils/evaluation.py'
    )
    
    parser.add_argument('summary', type=str,
                       help='Summary JSON file or directory containing summary_*.json files')
    parser.add_argument('--bpRNA-csv', type=str,
                       default='/projects/u6cg/jay/dissertations/data/bpRNA.csv',
                       help='Path to bpRNA.csv with ground truth')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for evaluation results')
    parser.add_argument('--shift', type=int, default=1,
                       help='Shift tolerance for pair matching (default: 1)')
    parser.add_argument('--use-base-pair-file', action='store_true',
                       help='Use base_pair.txt probability matrix instead of dot-bracket structure')
    
    args = parser.parse_args()
    
    summary_path = Path(args.summary)
    
    if summary_path.is_file():
        # Single summary file
        evaluate_summary(args.summary, args.bpRNA_csv, args.output, shift=args.shift, use_base_pair_file=args.use_base_pair_file)
    elif summary_path.is_dir():
        # Directory of summary files
        evaluate_all_summaries(args.summary, args.bpRNA_csv, args.output, shift=args.shift, use_base_pair_file=args.use_base_pair_file)
    else:
        print(f"[ERROR] Summary file/directory not found: {args.summary}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
