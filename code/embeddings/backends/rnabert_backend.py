"""RNABERT embedding backend."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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


MAX_LEN = 440
AMBIGUITY_MAP = {
    "B": "C",
    "D": "A",
    "H": "A",
    "K": "G",
    "M": "A",
    "N": "A",
    "R": "A",
    "S": "G",
    "V": "A",
    "W": "A",
    "Y": "C",
}


class ConfigDict(dict):
    """Dict wrapper with attribute access."""

    def __getattr__(self, key):
        return self[key]


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "rnabert",
        help="Generate per-layer RNABERT embeddings.",
    )
    parser.add_argument(
        "--external-root",
        default=os.environ.get("RNABERT_REPO", str(EXTERNAL_DIR / "RNABERT")),
        help="Path to the external RNABERT repository.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("RNABERT_CHECKPOINT", str(EXTERNAL_DIR / "RNABERT" / "bert_mul_2.pth")),
        help="Path to the RNABERT checkpoint.",
    )
    parser.add_argument(
        "--config-path",
        default=os.environ.get("RNABERT_CONFIG", str(EXTERNAL_DIR / "RNABERT" / "RNA_bert_config.json")),
        help="Path to the RNABERT config JSON.",
    )
    parser.add_argument(
        "--temp-dir",
        default="/tmp/rnabert_embeddings",
        help="Directory for temporary FASTA and text outputs.",
    )
    parser.add_argument(
        "--error-log",
        default=None,
        help="Optional CSV for per-sequence failures.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to run MLM_SFP.py.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Batch size passed to MLM_SFP.py.",
    )
    parser.set_defaults(run_backend=run_backend)
    return parser


def load_config(config_path: Path) -> ConfigDict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return ConfigDict(json.load(handle))


def normalize_sequence(sequence: str) -> str:
    sequence = sequence.strip()
    normalized = []
    for char in sequence:
        upper = char.upper()
        normalized.append(AMBIGUITY_MAP.get(upper, upper))
    invalid = sorted(set(normalized) - {"A", "C", "G", "U"})
    if invalid:
        raise ValueError(f"Unsupported nucleotide codes after normalization: {invalid}")
    return "".join(normalized)


def run_rnabert(
    external_root: Path,
    checkpoint: Path,
    input_fasta: Path,
    output_txt: Path,
    python_exe: str,
    batch_size: int,
) -> None:
    cmd = [
        python_exe,
        str(external_root / "MLM_SFP.py"),
        "--pretraining",
        str(checkpoint),
        "--data_embedding",
        str(input_fasta),
        "--embedding_output",
        str(output_txt),
        "--batch",
        str(batch_size),
    ]
    subprocess.run(cmd, cwd=external_root, check=True)


def parse_embedding(output_txt: Path, config: ConfigDict) -> np.ndarray:
    raw = output_txt.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Empty RNABERT output: {output_txt}")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = ast.literal_eval(raw)

    array = np.asarray(payload)
    expected_shape = (config.num_hidden_layers, array.shape[1], 120)
    if array.ndim != 3 or array.shape[0] != expected_shape[0] or array.shape[2] != expected_shape[2]:
        raise ValueError(f"Unexpected RNABERT output shape: {array.shape}")
    return array


def run_single_sequence(
    sequence: str,
    seq_id: str,
    config: ConfigDict,
    external_root: Path,
    checkpoint: Path,
    temp_dir: Path,
    python_exe: str,
    batch_size: int,
) -> np.ndarray:
    sequence = normalize_sequence(sequence)
    temp_dir.mkdir(parents=True, exist_ok=True)

    def write_fasta(path: Path, payload: str) -> None:
        path.write_text(f">rna\n{payload}", encoding="utf-8")

    def run_fragment(suffix: str, payload: str) -> np.ndarray:
        input_path = temp_dir / f"{seq_id}{suffix}.fa"
        output_path = temp_dir / f"{seq_id}{suffix}.txt"
        write_fasta(input_path, payload)
        try:
            run_rnabert(external_root, checkpoint, input_path, output_path, python_exe, batch_size)
            return parse_embedding(output_path, config)
        finally:
            if input_path.exists():
                input_path.unlink()
            if output_path.exists():
                output_path.unlink()

    if len(sequence) < MAX_LEN:
        return run_fragment("", sequence)

    front = run_fragment("_p1", sequence[: MAX_LEN - 1])
    back = run_fragment("_p2", sequence[MAX_LEN - 64 - 1 :])
    return np.concatenate([front[:, : 409 - 32, :], back[:, 32:, :]], axis=1)


def run_backend(args) -> None:
    metadata_csv = resolve_metadata_csv(args.dataset, args.metadata_csv)
    output_root = resolve_output_root("rnabert", args.dataset, args.output_root)
    external_root = require_existing_path(
        Path(args.external_root).expanduser().resolve(),
        "RNABERT repository",
    )
    require_existing_path(external_root / "MLM_SFP.py", "RNABERT MLM_SFP.py")
    checkpoint = require_existing_path(
        Path(args.checkpoint).expanduser().resolve(),
        "RNABERT checkpoint",
    )
    config_path = require_existing_path(
        Path(args.config_path).expanduser().resolve(),
        "RNABERT config",
    )
    config = load_config(config_path)
    temp_dir = Path(args.temp_dir).expanduser().resolve()
    layer_indices = list(range(config.num_hidden_layers))

    prepare_output_dirs(output_root, layer_indices)
    records = load_records(
        metadata_csv,
        ids_file=args.ids_file,
        start_idx=args.start_idx,
        limit=args.limit,
    )
    summarize_run(records, metadata_csv, output_root)

    errors: list[dict[str, object]] = []
    for record in tqdm(records, desc="rnabert"):
        if should_skip_sequence(output_root, record.seq_id, layer_indices, args.overwrite):
            continue
        try:
            embedding = run_single_sequence(
                sequence=record.sequence,
                seq_id=record.seq_id,
                config=config,
                external_root=external_root,
                checkpoint=checkpoint,
                temp_dir=temp_dir,
                python_exe=args.python_exe,
                batch_size=args.batch_size,
            )
            save_layer_outputs(
                output_root,
                record.seq_id,
                {idx: embedding[idx] for idx in layer_indices},
            )
        except Exception as exc:
            errors.append(
                {
                    "seq_id": record.seq_id,
                    "length": len(record.sequence),
                    "error": str(exc),
                }
            )

    if args.error_log and errors:
        import pandas as pd

        pd.DataFrame(errors).to_csv(args.error_log, index=False)
        print(f"error log: {args.error_log}")
