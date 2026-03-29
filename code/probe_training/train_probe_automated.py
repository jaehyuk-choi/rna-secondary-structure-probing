"""Train StructuralContactProbe on cached embeddings; main entry for sbatch_train_model.sh."""
import sys
import os
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, CURRENT_DIR)

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import random
from tqdm import tqdm

from models.bilinear_probe_model import BilinearContactProbe, StructuralContactProbe
from dataset_probe import RNABasepairDataset, rna_collate_fn
from utils.evaluation import compute_pair_metrics, precision_recall_f1
from experiment_config import ExperimentConfig, ROOT

# ============================================================
# Configuration
# ============================================================
META_CSV = os.path.join(ROOT, "data", "metadata", "bpRNA.csv")
SPLIT_CSV = os.path.join(ROOT, "data", "splits", "bpRNA_splits.csv")
CONTACT_DIR = os.path.join(ROOT, "data", "contact_maps", "bpRNA")

BATCH_SIZE = 4
NUM_EPOCHS = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Early stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0

# Sampling / weighting
USE_POS_WEIGHT = False
NEGATIVE_SAMPLING = True
NEG_RATIO = 1.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def compute_pos_neg_ratio(contact_dir, ids=None, use_weight=True):
    pos = 0
    neg = 0
    allowed = None
    if ids is not None:
        allowed = {f"{rid}_contact.npy" for rid in ids}

    files = [f for f in os.listdir(contact_dir) if f.endswith(".npy")]
    if allowed is not None:
        files = [f for f in files if f in allowed]

    for fname in files:
        C = np.load(os.path.join(contact_dir, fname))
        L = C.shape[0]
        upper = np.triu(np.ones((L, L), dtype=bool), 1)
        y = C[upper]
        pos += y.sum()
        neg += len(y) - y.sum()

    if use_weight:
        pos_weight = neg / pos
    else:
        pos_weight = neg / (pos + 1e-8)

    return pos, neg, pos_weight


def load_bprna_partitions(meta_csv: str, split_csv: str, max_len: int | None = 440):
    meta = pd.read_csv(meta_csv)
    meta["length"] = meta["sequence"].str.len()

    splits = pd.read_csv(split_csv)
    merged = splits.merge(meta[["id", "length"]], on="id", how="left")

    if max_len is not None:
        merged = merged[merged["length"] <= max_len]

    train_ids = merged[merged["partition"].isin(["TR0", "TR1"])]["id"].tolist()
    val_ids = merged[merged["partition"].isin(["VL0", "VL1"])]["id"].tolist()
    ts_ids = merged[merged["partition"].str.startswith("TS")]["id"].tolist()
    new_ids = merged[merged["partition"].str.lower() == "new"]["id"].tolist()

    print("\n=== bpRNA official split counts (after length filter) ===")
    print(f"Train (TR0+TR1): {len(train_ids)}")
    print(f"Val   (VL0+VL1): {len(val_ids)}")
    print(f"Test  (TS): {len(ts_ids)}")
    print(f"New        : {len(new_ids)}")
    print("=" * 30)

    return {
        "train": train_ids,
        "val": val_ids,
        "ts": ts_ids,
        "new": new_ids,
    }, merged


def filter_ids_by_embedding(ids: List[str], embedding_dir: str, embedding_suffix: str = ".npy") -> List[str]:
    filtered_ids = []
    missing_ids = []

    for seq_id in ids:
        emb_path = os.path.join(embedding_dir, f"{seq_id}{embedding_suffix}")
        if os.path.exists(emb_path):
            filtered_ids.append(seq_id)
        else:
            missing_ids.append(seq_id)

    if missing_ids:
        print(f"\n  embeddings missing: {len(missing_ids)} ids, ok: {len(filtered_ids)}")

    return filtered_ids


def get_embedding_dir(model: str, layer: int) -> str:
    base_dir = os.path.join(ROOT, "data", "embeddings")
    dataset_name = "bpRNA"

    dir_map = {
        "rnabert":  ("RNABERT",  "ArchiveII"),
        "ernie":    ("RNAErnie", "ArchiveII"),
        "rnafm":    ("RNAFM",    "ArchiveII"),
        "rinalmo":  ("RiNALMo",  "bpRNA"),
        "roberta":  ("RoBERTa",  "bpRNA"),
        "onehot":   ("Onehot",   "bpRNA"),
    }

    if model not in dir_map:
        raise ValueError(f"Unknown model: {model}")

    primary_name, fallback_dataset = dir_map[model]
    candidate = os.path.join(base_dir, primary_name, dataset_name, "by_layer", f"layer_{layer}")
    fallback = os.path.join(base_dir, primary_name, fallback_dataset, "by_layer", f"layer_{layer}")
    return candidate if os.path.isdir(candidate) else fallback


def train_probe(
    config: ExperimentConfig,
    train_ids: List[str],
    val_ids: List[str],
    run_dir: str,
) -> Dict:
    try:
        set_seed(config.seed)
        os.makedirs(run_dir, exist_ok=True)

        embedding_dir = get_embedding_dir(config.model, config.layer)
        print(f"\n  embeddings {config.model} layer {config.layer}: {embedding_dir}")

        train_ids_filtered = filter_ids_by_embedding(train_ids, embedding_dir)
        val_ids_filtered = filter_ids_by_embedding(val_ids, embedding_dir)

        if len(train_ids_filtered) == 0:
            raise ValueError(f"No training sequences with embeddings found in {embedding_dir}")

        print(f"\n  train {len(train_ids_filtered)}/{len(train_ids)}  val {len(val_ids_filtered)}/{len(val_ids)}")

        # Build datasets & loaders
        train_dataset = RNABasepairDataset(train_ids_filtered, embedding_dir, CONTACT_DIR)
        val_dataset = RNABasepairDataset(val_ids_filtered, embedding_dir, CONTACT_DIR)

        generator = torch.Generator()
        generator.manual_seed(config.seed)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=rna_collate_fn,
            generator=generator
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE,
            shuffle=False, collate_fn=rna_collate_fn
        )

        # Infer embedding dimension from first batch
        example_emb, _, _, _ = next(iter(train_loader))
        _, _, D = example_emb.shape

        probe = StructuralContactProbe(input_dim=D, proj_dim=config.k).to(DEVICE)

        # Class imbalance
        use_weight = not (USE_POS_WEIGHT and NEGATIVE_SAMPLING)
        pos, neg, pos_weight = compute_pos_neg_ratio(CONTACT_DIR, ids=train_ids, use_weight=use_weight)
        pos_weight_tensor = torch.tensor([pos_weight], device=DEVICE)

        print(f"=== Class imbalance (train only) ===")
        print(f"Positive pairs: {pos}")
        print(f"Negative pairs: {neg}")
        print(f"Computed pos_weight: {pos_weight:.2f}")
        print("=" * 30)

        # Loss & optimizer
        if USE_POS_WEIGHT:
            criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss(reduction="sum")

        optimizer = torch.optim.Adam(probe.parameters(), lr=LR)

        train_history = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_probe_state = None

        print("\n" + "=" * 80)
        print("Training")
        print("=" * 80)

        for epoch in range(1, NUM_EPOCHS + 1):
            epoch_start_time = time.time()
            probe.train()
            train_loss = 0.0
            train_pairs = 0

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
            for batch_embs, contacts_list, mask, ids in train_pbar:
                batch_embs = batch_embs.to(DEVICE)
                mask = mask.to(DEVICE)
                logits = probe(batch_embs, mask=mask)
                seq_losses = []

                for b_idx, contact in enumerate(contacts_list):
                    contact = contact.to(DEVICE)
                    L_contact = contact.shape[0]
                    L_emb = mask[b_idx].sum().item()
                    L_i = min(L_emb, L_contact)

                    logit_seq = logits[b_idx, :L_i, :L_i]
                    contact_truncated = contact[:L_i, :L_i]

                    triu = torch.triu(torch.ones(L_i, L_i, dtype=torch.bool, device=DEVICE), 1)
                    y_true_full = contact_truncated[triu]
                    y_pred_full = logit_seq[triu]

                    if y_true_full.numel() == 0:
                        continue

                    if NEGATIVE_SAMPLING:
                        pos_idx = (y_true_full == 1).nonzero(as_tuple=False).squeeze(1)
                        neg_idx = (y_true_full == 0).nonzero(as_tuple=False).squeeze(1)

                        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                            continue

                        num_pos = pos_idx.numel()
                        num_neg = int(num_pos * NEG_RATIO)

                        if neg_idx.numel() > num_neg:
                            perm = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:num_neg]
                            sampled_neg_idx = neg_idx[perm]
                        else:
                            sampled_neg_idx = neg_idx

                        selected_idx = torch.cat([pos_idx, sampled_neg_idx], dim=0)
                        y_true = y_true_full[selected_idx]
                        y_pred_logits = y_pred_full[selected_idx]
                    else:
                        y_true = y_true_full
                        y_pred_logits = y_pred_full

                    if y_true.numel() == 0:
                        continue

                    seq_loss = criterion(y_pred_logits, y_true)
                    seq_losses.append(seq_loss)
                    train_pairs += y_true.numel()

                if len(seq_losses) == 0:
                    continue

                loss = torch.stack(seq_losses).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = train_loss / max(1, train_pairs) if train_pairs > 0 else 0.0

            # Validation
            probe.eval()
            val_loss = 0.0
            val_pairs = 0

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]")
            with torch.no_grad():
                for batch_embs, contacts_list, mask, ids in val_pbar:
                    batch_embs = batch_embs.to(DEVICE)
                    mask = mask.to(DEVICE)
                    logits = probe(batch_embs, mask=mask)

                    for b_idx, contact in enumerate(contacts_list):
                        contact = contact.to(DEVICE)
                        L_contact = contact.shape[0]
                        L_emb = mask[b_idx].sum().item()
                        L_i = min(L_emb, L_contact)

                        logit_seq = logits[b_idx, :L_i, :L_i]
                        contact_truncated = contact[:L_i, :L_i]

                        triu = torch.triu(torch.ones(L_i, L_i, dtype=torch.bool, device=DEVICE), 1)
                        y_true_full = contact_truncated[triu]
                        y_pred_logits_full = logit_seq[triu]

                        if y_true_full.numel() > 0:
                            if NEGATIVE_SAMPLING:
                                pos_idx = (y_true_full == 1).nonzero(as_tuple=False).squeeze(1)
                                neg_idx = (y_true_full == 0).nonzero(as_tuple=False).squeeze(1)

                                if pos_idx.numel() > 0 and neg_idx.numel() > 0:
                                    num_pos = pos_idx.numel()
                                    num_neg = int(num_pos * NEG_RATIO)

                                    if neg_idx.numel() > num_neg:
                                        perm = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:num_neg]
                                        sampled_neg_idx = neg_idx[perm]
                                    else:
                                        sampled_neg_idx = neg_idx

                                    selected_idx = torch.cat([pos_idx, sampled_neg_idx], dim=0)
                                    y_true = y_true_full[selected_idx]
                                    y_pred_logits = y_pred_logits_full[selected_idx]
                                else:
                                    y_true = y_true_full
                                    y_pred_logits = y_pred_logits_full
                            else:
                                y_true = y_true_full
                                y_pred_logits = y_pred_logits_full

                            seq_loss = criterion(y_pred_logits, y_true)
                            val_loss += seq_loss.item()
                            val_pairs += y_true.numel()

                    val_pbar.set_postfix({"val_loss": f"{val_loss/max(1,val_pairs):.4f}"})

            avg_val_loss = val_loss / max(1, val_pairs) if val_pairs > 0 else float('inf')
            epoch_time = time.time() - epoch_start_time

            train_history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": LR,
                "time_sec": epoch_time,
                "seed": config.seed,
                "model": config.model,
                "layer": config.layer,
                "k": config.k,
            })

            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Time={epoch_time:.1f}s")

            # Early stopping on validation loss
            if avg_val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                best_probe_state = probe.state_dict().copy()
                print(f"  New best validation loss: {best_val_loss:.4f} (epoch {epoch})")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
                    if best_probe_state is not None:
                        probe.load_state_dict(best_probe_state)
                    break

        # Persist artefacts
        os.makedirs(run_dir, exist_ok=True)

        checkpoint_path = os.path.join(run_dir, "best.pt")
        if best_probe_state is not None:
            torch.save({
                "model_state_dict": best_probe_state,
                "input_dim": D,
                "proj_dim": config.k,
                "model": config.model,
                "layer": config.layer,
                "seed": config.seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            print(f"\n  Best checkpoint saved to: {checkpoint_path}")

        history_df = pd.DataFrame(train_history)
        history_path = os.path.join(run_dir, "train_history.csv")
        history_df.to_csv(history_path, index=False)
        print(f"  Training history saved to: {history_path}")

        run_config_df = pd.DataFrame([{
            "model": config.model,
            "layer": config.layer,
            "k": config.k,
            "seed": config.seed,
            "max_epoch": NUM_EPOCHS,
            "patience": EARLY_STOPPING_PATIENCE,
            "optimizer": "Adam",
            "lr": LR,
            "loss_name": "BCEWithLogitsLoss",
            "pos_weight": pos_weight,
            "decoding_rule": "greedy_max1" if NEGATIVE_SAMPLING else "full",
            "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }])
        run_config_path = os.path.join(run_dir, "run_config.csv")
        run_config_df.to_csv(run_config_path, index=False)
        print(f"  Run config saved to: {run_config_path}")

        best_checkpoint_df = pd.DataFrame([{
            "model": config.model,
            "layer": config.layer,
            "k": config.k,
            "seed": config.seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "checkpoint_path": checkpoint_path,
        }])
        best_checkpoint_path = os.path.join(run_dir, "best_checkpoint.csv")
        best_checkpoint_df.to_csv(best_checkpoint_path, index=False)
        print(f"  Best checkpoint info saved to: {best_checkpoint_path}")

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "checkpoint_path": checkpoint_path,
            "train_history_path": history_path,
            "run_config_path": run_config_path,
            "best_checkpoint_path": best_checkpoint_path,
        }
    except Exception as e:
        print(f"\n  training error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train probe for RNA contact prediction")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--layer", type=int, required=True, help="Layer number")
    parser.add_argument("--k", type=int, required=True, help="Projection dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = ExperimentConfig(
        model=args.model,
        layer=args.layer,
        k=args.k,
        seed=args.seed,
    )

    partitions, _ = load_bprna_partitions(META_CSV, SPLIT_CSV, max_len=440)

    summary = train_probe(
        config=config,
        train_ids=partitions["train"],
        val_ids=partitions["val"],
        run_dir=config.run_dir,
    )

    print("\n" + "=" * 80)
    print("Training done")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Best validation loss: {summary['best_val_loss']:.4f}")
    print(f"Checkpoint: {summary['checkpoint_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
