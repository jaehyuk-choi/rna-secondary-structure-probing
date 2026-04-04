#!/usr/bin/env python3
"""Generate per-sequence, per-layer embeddings from the repository metadata tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from embeddings.common import add_common_arguments
from embeddings.backends import (
    ernie_rna_backend,
    onehot_backend,
    rnabert_backend,
    rinalmo_backend,
    rnafm_backend,
    roberta_backend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate embedding files under data/embeddings/<MODEL>/<dataset>/by_layer.",
    )
    subparsers = parser.add_subparsers(dest="model", required=True)

    backends = [
        ernie_rna_backend,
        onehot_backend,
        rnabert_backend,
        rinalmo_backend,
        rnafm_backend,
        roberta_backend,
    ]
    for backend in backends:
        subparser = backend.register_parser(subparsers)
        add_common_arguments(subparser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.run_backend(args)


if __name__ == "__main__":
    main()

