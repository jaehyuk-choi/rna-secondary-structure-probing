"""One-hot embedding backend."""

from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from embeddings.common import (
    load_records,
    prepare_output_dirs,
    resolve_metadata_csv,
    resolve_output_root,
    save_layer_outputs,
    should_skip_sequence,
    summarize_run,
)


BASE_TO_INDEX = {"A": 0, "U": 1, "G": 2, "C": 3}


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "onehot",
        help="Generate one-hot nucleotide embeddings.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer index used for the one-hot baseline.",
    )
    parser.set_defaults(run_backend=run_backend)
    return parser


def encode_sequence(sequence: str) -> np.ndarray:
    embedding = np.zeros((len(sequence), 4), dtype=np.float32)
    for idx, base in enumerate(sequence.upper()):
        if base in BASE_TO_INDEX:
            embedding[idx, BASE_TO_INDEX[base]] = 1.0
        else:
            embedding[idx, :] = 0.25
    return embedding


def run_backend(args) -> None:
    metadata_csv = resolve_metadata_csv(args.dataset, args.metadata_csv)
    output_root = resolve_output_root("onehot", args.dataset, args.output_root)
    layer_indices = [args.layer]
    prepare_output_dirs(output_root, layer_indices)

    records = load_records(
        metadata_csv,
        ids_file=args.ids_file,
        start_idx=args.start_idx,
        limit=args.limit,
    )
    summarize_run(records, metadata_csv, output_root)

    for record in tqdm(records, desc="onehot"):
        if should_skip_sequence(output_root, record.seq_id, layer_indices, args.overwrite):
            continue
        save_layer_outputs(
            output_root,
            record.seq_id,
            {args.layer: encode_sequence(record.sequence)},
        )

