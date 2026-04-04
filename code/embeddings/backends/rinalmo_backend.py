"""RiNALMo embedding backend."""

from __future__ import annotations

import argparse
import importlib.util
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


class MockFlashAttn:
    @staticmethod
    def flash_attn_varlen_qkvpacked_func(*args, **kwargs):
        raise NotImplementedError("flash_attn is not available")

    @staticmethod
    def flash_attn_qkvpacked_func(*args, **kwargs):
        raise NotImplementedError("flash_attn is not available")

    class layers:
        class rotary:
            class RotaryEmbedding:
                def __init__(self, *args, **kwargs):
                    pass

    class bert_padding:
        @staticmethod
        def unpad_input(*args, **kwargs):
            raise NotImplementedError("flash_attn is not available")

        @staticmethod
        def pad_input(*args, **kwargs):
            raise NotImplementedError("flash_attn is not available")


class RiNALMoWithLayerOutputs:
    """Wrap a RiNALMo model and capture intermediate states through hooks."""

    def __init__(self, model):
        self.model = model
        self.layer_outputs = []
        self.hooks = []

        if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
            for layer_idx, block in enumerate(model.transformer.blocks):
                self.hooks.append(block.register_forward_hook(self._make_hook(layer_idx)))

    def _make_hook(self, layer_idx):
        def hook(_module, _inputs, outputs):
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            while len(self.layer_outputs) <= layer_idx:
                self.layer_outputs.append(None)
            self.layer_outputs[layer_idx] = hidden_states.detach()

        return hook

    def __call__(self, tokens):
        import torch

        self.layer_outputs = []
        with torch.no_grad():
            outputs = self.model(tokens)
            if hasattr(self.model, "embedding"):
                embedding = self.model.embedding(tokens)
                if hasattr(self.model, "token_dropout"):
                    embedding = self.model.token_dropout(embedding, tokens)
                self.layer_outputs.insert(0, embedding.detach())
            if "representation" in outputs:
                while len(self.layer_outputs) <= len(self.model.transformer.blocks):
                    self.layer_outputs.append(None)
                self.layer_outputs[-1] = outputs["representation"].detach()
        return outputs

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "rinalmo",
        help="Generate per-layer RiNALMo embeddings.",
    )
    parser.add_argument(
        "--external-root",
        default=os.environ.get("RINALMO_REPO", str(EXTERNAL_DIR / "RiNALMo")),
        help="Path to the external RiNALMo repository.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("RINALMO_CHECKPOINT", str(EXTERNAL_DIR / "RiNALMo" / "weights" / "rinalmo_giga_pretrained.pt")),
        help="Path to the RiNALMo checkpoint.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device. Defaults to cuda when available.",
    )
    parser.set_defaults(run_backend=run_backend)
    return parser


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def install_flash_attention_stub() -> None:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        sys.modules["flash_attn"] = MockFlashAttn()
        sys.modules["flash_attn.layers"] = MockFlashAttn.layers
        sys.modules["flash_attn.layers.rotary"] = MockFlashAttn.layers.rotary
        sys.modules["flash_attn.bert_padding"] = MockFlashAttn.bert_padding


def run_backend(args) -> None:
    import numpy as np
    import torch

    external_root = require_existing_path(
        Path(args.external_root).expanduser().resolve(),
        "RiNALMo repository",
    )
    if str(external_root) not in sys.path:
        sys.path.insert(0, str(external_root))

    install_flash_attention_stub()

    init_path = external_root / "rinalmo" / "__init__.py"
    pretrained_path = external_root / "rinalmo" / "pretrained.py"
    if init_path.exists() and "rinalmo" not in sys.modules:
        load_module("rinalmo", init_path)
    if pretrained_path.exists() and "rinalmo.pretrained" not in sys.modules:
        load_module("rinalmo.pretrained", pretrained_path)

    from rinalmo.config import model_config
    from rinalmo.data.alphabet import Alphabet
    from rinalmo.model.model import RiNALMo
    from rinalmo.pretrained import get_pretrained_model

    metadata_csv = resolve_metadata_csv(args.dataset, args.metadata_csv)
    output_root = resolve_output_root("rinalmo", args.dataset, args.output_root)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = model_config("giga")
    if isinstance(sys.modules.get("flash_attn"), MockFlashAttn):
        config["model"]["transformer"]["use_flash_attn"] = False

    model = RiNALMo(config)
    alphabet = Alphabet(**config["alphabet"])
    checkpoint = require_existing_path(
        Path(args.checkpoint).expanduser().resolve(),
        "RiNALMo checkpoint",
    )

    original_load = torch.load

    def patched_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    try:
        torch.load = patched_load
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=device)
            state_dict = state.get("model", state.get("state_dict", state)) if isinstance(state, dict) else state
            model.load_state_dict(state_dict, strict=False)
        else:
            model, alphabet = get_pretrained_model(model_name="giga-v1")
    finally:
        torch.load = original_load

    model = model.to(device)
    model.eval()
    wrapper = RiNALMoWithLayerOutputs(model)
    layer_indices = list(range(len(model.transformer.blocks) + 1))
    prepare_output_dirs(output_root, layer_indices)

    records = load_records(
        metadata_csv,
        ids_file=args.ids_file,
        start_idx=args.start_idx,
        limit=args.limit,
    )
    summarize_run(records, metadata_csv, output_root)

    with torch.no_grad():
        for record in tqdm(records, desc="rinalmo"):
            if should_skip_sequence(output_root, record.seq_id, layer_indices, args.overwrite):
                continue

            tokens = torch.tensor(alphabet.batch_tokenize([record.sequence]), dtype=torch.int64, device=device)
            wrapper(tokens)

            layer_outputs = {}
            for layer_idx in layer_indices:
                tensor = wrapper.layer_outputs[layer_idx]
                if tensor is None:
                    raise ValueError(f"Missing RiNALMo layer output: layer_{layer_idx}")
                if tensor.dim() == 3:
                    tensor = tensor.squeeze(0)
                if tensor.shape[0] > 2:
                    tensor = tensor[1:-1, :]
                elif tensor.shape[0] > 1:
                    tensor = tensor[1:, :]
                layer_outputs[layer_idx] = np.asarray(tensor.cpu().numpy())

            save_layer_outputs(output_root, record.seq_id, layer_outputs)

    wrapper.close()
