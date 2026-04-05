#!/usr/bin/env python3
"""VL0 pair-score densities: sigmoid probe logits vs GT contacts, one panel per model."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR / "probe_inference"))
sys.path.insert(0, str(CODE_DIR / "probe_training"))

from experiment_config import MODEL_CONFIGS  # noqa: E402
from generate_base_pairs import load_probe_matrix  # noqa: E402

# kde gets expensive past ~50k points; subsample for fitting only
KDE_MAX_SAMPLES = 50_000

BEST_CONFIG_PATH = REPO_ROOT / "configs" / "best_config_val_f1.csv"
META_CSV = REPO_ROOT / "data" / "metadata" / "bpRNA.csv"
SPLIT_CSV = REPO_ROOT / "data" / "splits" / "bpRNA_splits.csv"
CONTACT_DIR = REPO_ROOT / "data" / "contact_maps" / "bpRNA"
OUTPUT_DEFAULT = REPO_ROOT / "figures" / "main" / "vl0_score_distributions.png"

MODEL_ORDER = ["ernie", "roberta", "rnafm", "rinalmo", "onehot", "rnabert"]
MODEL_LABELS = {
    "ernie": "ERNIE-RNA",
    "roberta": "RoBERTa",
    "rnafm": "RNA-FM",
    "rinalmo": "RiNALMo",
    "onehot": "One-hot",
    "rnabert": "RNABERT",
}


def load_best_config_rows() -> dict[str, dict]:
    with open(BEST_CONFIG_PATH) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    legacy = "selected_layer" in rows[0]
    cfg: dict[str, dict] = {}
    for r in rows:
        m = r.get("model", "").strip()
        if not m:
            continue
        if legacy:
            cfg[m] = {
                "layer": int(r["selected_layer"]),
                "k": int(r["selected_k"]),
                "threshold": float(r["selected_best_threshold"]),
                "decoding_mode": r["selected_decoding_mode"],
            }
        else:
            cfg[m] = {
                "layer": int(r["layer"]),
                "k": int(r["k"]),
                "threshold": float(r["threshold"]),
                "decoding_mode": r.get("decoding_mode", ""),
            }
    return cfg


def load_vl0_ids(max_len: int | None = 440) -> list[str]:
    lengths: dict[str, int] = {}
    with open(META_CSV) as f:
        for r in csv.DictReader(f):
            sid = r.get("id", "")
            if not sid:
                continue
            if "length" in r and r["length"] not in ("", None):
                try:
                    L = int(float(r["length"]))
                except ValueError:
                    L = len(r.get("sequence", "") or "")
            elif "sequence" in r:
                L = len(r["sequence"] or "")
            else:
                continue
            lengths[sid] = L

    out: list[str] = []
    with open(SPLIT_CSV) as f:
        for r in csv.DictReader(f):
            if r.get("partition") != "VL0":
                continue
            sid = r.get("id", "")
            if not sid or sid not in lengths:
                continue
            if max_len is not None and lengths[sid] > max_len:
                continue
            out.append(sid)
    return out


def embedding_dir_for(model: str, layer: int) -> Path:
    base = MODEL_CONFIGS[model]["base_path"]
    return Path(base) / f"layer_{layer}"


def checkpoint_path_for(model: str, layer: int, k: int, seed: int = 42) -> Path:
    return (
        REPO_ROOT
        / "results"
        / "outputs"
        / model
        / f"layer_{layer}"
        / f"k_{k}"
        / f"seed_{seed}"
        / "best.pt"
    )


def pair_probs(emb_t: torch.Tensor, B: torch.Tensor, device: torch.device) -> torch.Tensor:
    B = B.to(device)
    z = torch.matmul(emb_t, B.t())
    logits = torch.matmul(z, z.t())
    return torch.sigmoid(logits)


def collect_pos_neg_scores(
    model: str,
    layer: int,
    k_cfg: int,
    vl0_ids: list[str],
    device: torch.device,
    rng: np.random.Generator,
    neg_multiple: float,
) -> tuple[np.ndarray, np.ndarray]:
    ckpt = checkpoint_path_for(model, layer, k_cfg)
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))

    emb_root = embedding_dir_for(model, layer)
    if not emb_root.is_dir():
        raise FileNotFoundError(str(emb_root))

    with contextlib.redirect_stdout(io.StringIO()):
        B, _, _ = load_probe_matrix(str(ckpt), device=str(device))
    B = B.to(device)

    pos_chunks: list[np.ndarray] = []
    neg_chunks: list[np.ndarray] = []

    for sid in vl0_ids:
        emb_path = emb_root / f"{sid}.npy"
        ct_path = CONTACT_DIR / f"{sid}_contact.npy"
        if not emb_path.exists() or not ct_path.exists():
            continue

        emb = np.load(emb_path)
        M = np.load(ct_path)
        if emb.ndim != 2 or M.ndim != 2:
            continue
        L = min(emb.shape[0], M.shape[0], M.shape[1])
        emb = emb[:L].astype(np.float32)
        M = M[:L, :L]

        emb_t = torch.from_numpy(emb).to(device)
        p = pair_probs(emb_t, B, device).detach().cpu().numpy()

        ui, uj = np.triu_indices(L, k=1)
        scores = p[ui, uj]
        gt = M[ui, uj]
        pos_chunks.append(scores[gt > 0.5])
        neg_chunks.append(scores[gt <= 0.5])

    if not pos_chunks:
        return np.array([]), np.array([])

    pos_all = np.concatenate(pos_chunks)
    neg_all = np.concatenate(neg_chunks)

    cap = int(neg_multiple * len(pos_all))
    if len(neg_all) > cap:
        neg_all = neg_all[rng.choice(len(neg_all), size=cap, replace=False)]

    return pos_all, neg_all


def kde_on_grid(
    x_grid: np.ndarray, samples: np.ndarray, rng: np.random.Generator
) -> np.ndarray | None:
    if samples.size < 2:
        return None
    s = samples
    if s.size > KDE_MAX_SAMPLES:
        s = rng.choice(s, size=KDE_MAX_SAMPLES, replace=False)
    s = np.clip(s, 1e-9, 1.0 - 1e-9)
    try:
        y = gaussian_kde(s)(x_grid)
    except (np.linalg.LinAlgError, ValueError):
        return None
    area = getattr(np, "trapezoid", np.trapz)(y, x_grid)
    if area > 0:
        y /= area
    return y


def draw_panel(
    ax,
    model: str,
    pos: np.ndarray,
    neg: np.ndarray,
    tau: float,
    layer: int,
    k: int,
    decoding_mode: str,
    rng: np.random.Generator,
) -> None:
    x = np.linspace(0.0, 1.0, 512)

    yp = kde_on_grid(x, pos, rng) if pos.size else None
    yn = kde_on_grid(x, neg, rng) if neg.size else None

    if yp is not None:
        ax.plot(x, yp, color="#2ecc71", lw=2.0, label="M=1")
    if yn is not None:
        ax.plot(x, yn, color="#7f8c8d", lw=2.0, label="M=0")

    ax.axvline(tau, color="#c0392b", ls="--", lw=1.5, label=f"τ*={tau:.2f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"Probe score $p_{ij}$")
    ax.set_ylabel("Density (normalized)")
    dm = decoding_mode if len(decoding_mode) < 28 else decoding_mode[:25] + "…"
    ax.set_title(f"{MODEL_LABELS.get(model, model)}\nL={layer}, k={k}, {dm}")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.25)


def main() -> int:
    p = argparse.ArgumentParser(description="VL0 score densities by model (best val config).")
    p.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-len", type=int, default=440)
    p.add_argument("--no-max-len", action="store_true")
    p.add_argument("--max-seq", type=int, default=None)
    p.add_argument("--neg-multiple", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not BEST_CONFIG_PATH.exists():
        print(BEST_CONFIG_PATH, "not found", file=sys.stderr)
        return 1

    cfg = load_best_config_rows()
    device = torch.device(args.device)
    max_len = None if args.no_max_len else args.max_len
    ids = load_vl0_ids(max_len=max_len)
    if args.max_seq is not None:
        ids = ids[: args.max_seq]

    rng = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), sharey=False)
    flat = axes.flatten()

    for ax, name in zip(flat, MODEL_ORDER):
        if name not in cfg:
            ax.set_title(f"{name}\n(no row in config)")
            continue
        c = cfg[name]
        layer, k, tau = c["layer"], c["k"], c["threshold"]
        dm = c.get("decoding_mode", "")
        try:
            pos, neg = collect_pos_neg_scores(
                name, layer, k, ids, device, rng, args.neg_multiple
            )
        except FileNotFoundError as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=8)
            ax.set_axis_off()
            continue
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=8)
            ax.set_axis_off()
            continue

        draw_panel(ax, name, pos, neg, tau, layer, k, dm, rng)

    plt.suptitle("VL0 probe score distributions", fontsize=12, y=1.01)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
