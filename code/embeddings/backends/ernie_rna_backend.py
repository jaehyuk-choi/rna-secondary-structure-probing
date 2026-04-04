"""ERNIE-RNA embedding backend."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

from embeddings.common import (
    EXTERNAL_DIR,
    load_records,
    prepare_output_dirs,
    require_existing_path,
    resolve_metadata_csv,
    resolve_output_root,
    save_layer_outputs,
    should_skip_sequence,
    summarize_run,
)


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "ernie",
        help="Generate per-layer ERNIE-RNA embeddings.",
    )
    parser.add_argument(
        "--external-root",
        default=os.environ.get("ERNIE_RNA_REPO", str(EXTERNAL_DIR / "ERNIE-RNA")),
        help="Path to the external ERNIE-RNA repository.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "ERNIE_RNA_CHECKPOINT",
            str(EXTERNAL_DIR / "ERNIE-RNA" / "checkpoint" / "ERNIE-RNA_checkpoint" / "ERNIE-RNA_pretrain.pt"),
        ),
        help="Path to the ERNIE-RNA checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device.",
    )
    parser.set_defaults(run_backend=run_backend)
    return parser


def run_backend(args) -> None:
    import numpy as np
    import torch

    external_root = require_existing_path(
        Path(args.external_root).expanduser().resolve(),
        "ERNIE-RNA repository",
    )
    require_existing_path(
        Path(args.checkpoint).expanduser().resolve(),
        "ERNIE-RNA checkpoint",
    )
    require_existing_path(external_root / "src" / "dict", "ERNIE-RNA token dictionary")
    if str(external_root) not in sys.path:
        sys.path.insert(0, str(external_root))

    import extract_embedding

    metadata_csv = resolve_metadata_csv(args.dataset, args.metadata_csv)
    output_root = resolve_output_root("ernie", args.dataset, args.output_root)
    layer_indices = list(range(12))
    prepare_output_dirs(output_root, layer_indices)

    records = load_records(
        metadata_csv,
        ids_file=args.ids_file,
        start_idx=args.start_idx,
        limit=args.limit,
    )
    summarize_run(records, metadata_csv, output_root)

    original_load = torch.load

    def patched_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    try:
        torch.load = patched_load
        for record in tqdm(records, desc="ernie"):
            if should_skip_sequence(output_root, record.seq_id, layer_indices, args.overwrite):
                continue

            embedding = extract_embedding.extract_embedding_of_ernierna(
                [record.sequence],
                if_cls=False,
                arg_overrides={"data": str(external_root / "src" / "dict")},
                pretrained_model_path=args.checkpoint,
                device=args.device,
            )
            array = np.squeeze(embedding, axis=0)[:, 1:-1, :]
            save_layer_outputs(output_root, record.seq_id, {idx: array[idx] for idx in layer_indices})
    finally:
        torch.load = original_load
