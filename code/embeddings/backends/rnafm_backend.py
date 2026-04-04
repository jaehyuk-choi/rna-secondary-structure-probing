"""RNA-FM embedding backend."""

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
        "rnafm",
        help="Generate per-layer RNA-FM embeddings.",
    )
    parser.add_argument(
        "--external-root",
        default=os.environ.get("RNAFM_REPO", str(EXTERNAL_DIR / "RNA-FM")),
        help="Path to the external RNA-FM repository.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("RNAFM_CHECKPOINT", str(EXTERNAL_DIR / "RNA-FM" / "RNA-FM_pretrained.pth")),
        help="Path to the RNA-FM checkpoint.",
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
        "RNA-FM repository",
    )
    require_existing_path(
        Path(args.checkpoint).expanduser().resolve(),
        "RNA-FM checkpoint",
    )
    if str(external_root) not in sys.path:
        sys.path.insert(0, str(external_root))

    from fm.pretrained import rna_fm_t12

    metadata_csv = resolve_metadata_csv(args.dataset, args.metadata_csv)
    output_root = resolve_output_root("rnafm", args.dataset, args.output_root)
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
        model, alphabet = rna_fm_t12(model_location=args.checkpoint)
    finally:
        torch.load = original_load

    model = model.to(args.device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    with torch.no_grad():
        for record in tqdm(records, desc="rnafm"):
            if should_skip_sequence(output_root, record.seq_id, layer_indices, args.overwrite):
                continue

            _, _, tokens = batch_converter([(record.seq_id, record.sequence)])
            result = model(tokens.to(args.device), repr_layers=list(range(13)))

            layer_outputs = {}
            for layer_idx in layer_indices:
                layer_array = result["representations"][layer_idx + 1].squeeze(0).cpu().numpy()
                layer_outputs[layer_idx] = np.asarray(layer_array)

            save_layer_outputs(output_root, record.seq_id, layer_outputs)
