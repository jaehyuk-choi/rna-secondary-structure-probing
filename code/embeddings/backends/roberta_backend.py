"""RoBERTa embedding backend."""

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


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "roberta",
        help="Generate per-layer RoBERTa embeddings.",
    )
    parser.add_argument(
        "--model-name",
        default="roberta-base",
        help="Hugging Face model identifier.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max length.",
    )
    parser.set_defaults(run_backend=run_backend)
    return parser


def _collect_nucleotide_embeddings(tokens, layer_array: np.ndarray, original_length: int, pad_token: str | None) -> np.ndarray:
    token_index = 1  # skip <s>
    embeddings = []

    for _ in range(original_length):
        if token_index >= len(tokens) or token_index >= layer_array.shape[0]:
            break

        embeddings.append(layer_array[token_index])
        token_index += 1

        while token_index < len(tokens):
            token = tokens[token_index]
            if token.startswith("</") or token in {"<s>", "</s>", "</w>", pad_token}:
                token_index += 1
                continue
            break

    if not embeddings:
        return np.zeros((original_length, layer_array.shape[1]), dtype=layer_array.dtype)

    if len(embeddings) < original_length:
        embeddings.extend([embeddings[-1]] * (original_length - len(embeddings)))
    elif len(embeddings) > original_length:
        embeddings = embeddings[:original_length]

    return np.asarray(embeddings)


def run_backend(args) -> None:
    import torch
    from transformers import RobertaModel, RobertaTokenizer

    metadata_csv = resolve_metadata_csv(args.dataset, args.metadata_csv)
    output_root = resolve_output_root("roberta", args.dataset, args.output_root)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaModel.from_pretrained(args.model_name, output_hidden_states=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    layer_indices = list(range(model.config.num_hidden_layers))
    prepare_output_dirs(output_root, layer_indices)

    records = load_records(
        metadata_csv,
        ids_file=args.ids_file,
        start_idx=args.start_idx,
        limit=args.limit,
    )
    summarize_run(records, metadata_csv, output_root)

    with torch.no_grad():
        for record in tqdm(records, desc="roberta"):
            if should_skip_sequence(output_root, record.seq_id, layer_indices, args.overwrite):
                continue

            encoded = tokenizer(
                " ".join(record.sequence),
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

            layer_outputs = {}
            for layer_idx, hidden_state in enumerate(outputs.hidden_states[1:]):
                layer_array = hidden_state.squeeze(0).cpu().numpy()
                final_array = _collect_nucleotide_embeddings(
                    tokens=tokens,
                    layer_array=layer_array,
                    original_length=len(record.sequence),
                    pad_token=tokenizer.pad_token,
                )
                layer_outputs[layer_idx] = final_array

            save_layer_outputs(output_root, record.seq_id, layer_outputs)

