"""Shared helpers for embedding extraction scripts."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("RNA_PROBE_DATA_ROOT", REPO_ROOT))
METADATA_DIR = DATA_ROOT / "data" / "metadata"
EMBEDDINGS_DIR = DATA_ROOT / "data" / "embeddings"
EXTERNAL_DIR = REPO_ROOT / "external"

MODEL_DIR_NAMES = {
    "ernie": "RNAErnie",
    "onehot": "Onehot",
    "rnabert": "RNABERT",
    "rnafm": "RNAFM",
    "rinalmo": "RiNALMo",
    "roberta": "RoBERTa",
}

DEFAULT_METADATA = {
    "ArchiveII": METADATA_DIR / "ArchiveII.csv",
    "bpRNA": METADATA_DIR / "bpRNA.csv",
}


@dataclass(frozen=True)
class SequenceRecord:
    """Minimal sequence record used by the embedding generators."""

    seq_id: str
    sequence: str


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=sorted(DEFAULT_METADATA),
        default="bpRNA",
        help="Dataset name used to resolve default metadata and output paths.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=None,
        help="Override the metadata CSV. Defaults to data/metadata/<dataset>.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override the output root. Defaults to data/embeddings/<MODEL>/<dataset>/by_layer.",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default=None,
        help="Optional text file with one sequence id per line.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start offset in the filtered metadata table.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of sequences to process after filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files instead of skipping them.",
    )


def resolve_metadata_csv(dataset: str, metadata_csv: str | None) -> Path:
    if metadata_csv is not None:
        return Path(metadata_csv).expanduser().resolve()
    return DEFAULT_METADATA[dataset]


def resolve_output_root(model_key: str, dataset: str, output_root: str | None) -> Path:
    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    return EMBEDDINGS_DIR / MODEL_DIR_NAMES[model_key] / dataset / "by_layer"


def read_sequence_table(metadata_csv: Path):
    import pandas as pd

    df = pd.read_csv(metadata_csv)
    if "id" not in df.columns:
        raise ValueError(f"Missing 'id' column in {metadata_csv}")

    if "sequence" not in df.columns:
        if "seq" in df.columns:
            df = df.rename(columns={"seq": "sequence"})
        else:
            raise ValueError(f"Missing 'sequence' column in {metadata_csv}")

    return df[["id", "sequence"]].copy()


def load_records(
    metadata_csv: Path,
    ids_file: str | None = None,
    start_idx: int = 0,
    limit: int | None = None,
) -> list[SequenceRecord]:
    table = read_sequence_table(metadata_csv)

    if ids_file is not None:
        with open(ids_file, "r", encoding="utf-8") as handle:
            keep_ids = {line.strip() for line in handle if line.strip()}
        table = table[table["id"].isin(keep_ids)]

    if start_idx:
        table = table.iloc[start_idx:]
    if limit is not None:
        table = table.iloc[:limit]

    return [
        SequenceRecord(seq_id=str(row.id), sequence=str(row.sequence))
        for row in table.itertuples(index=False)
    ]


def prepare_output_dirs(output_root: Path, layer_indices: Iterable[int]) -> None:
    for layer_idx in layer_indices:
        (output_root / f"layer_{layer_idx}").mkdir(parents=True, exist_ok=True)


def layer_output_path(output_root: Path, layer_idx: int, seq_id: str) -> Path:
    return output_root / f"layer_{layer_idx}" / f"{seq_id}.npy"


def layer_exists(output_root: Path, layer_idx: int, seq_id: str) -> bool:
    return layer_output_path(output_root, layer_idx, seq_id).exists()


def should_skip_sequence(
    output_root: Path,
    seq_id: str,
    expected_layers: Iterable[int],
    overwrite: bool,
) -> bool:
    if overwrite:
        return False
    return all(layer_exists(output_root, layer_idx, seq_id) for layer_idx in expected_layers)


def save_layer_outputs(output_root: Path, seq_id: str, layer_outputs: dict[int, np.ndarray]) -> None:
    for layer_idx, array in layer_outputs.items():
        np.save(layer_output_path(output_root, layer_idx, seq_id), array)


def summarize_run(records: list[SequenceRecord], metadata_csv: Path, output_root: Path) -> None:
    print(f"metadata: {metadata_csv}")
    print(f"records : {len(records)}")
    print(f"output  : {output_root}")


def require_existing_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path
