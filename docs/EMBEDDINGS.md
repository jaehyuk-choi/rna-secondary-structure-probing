# Embedding Generation

This repository expects per-sequence `.npy` embeddings under:

```text
data/embeddings/<MODEL>/<DATASET>/by_layer/layer_<N>/<seq_id>.npy
```

The extraction entry point is:

```bash
python code/embeddings/generate_embeddings.py <model> --dataset bpRNA
```

External upstream repositories can be prepared with:

```bash
bash scripts/setup_external_repos.sh
bash scripts/check_external_assets.sh
```

Supported models:

- `onehot`
- `roberta`
- `rnafm`
- `ernie`
- `rnabert`
- `rinalmo`

Examples:

```bash
python code/embeddings/generate_embeddings.py onehot --dataset bpRNA

python code/embeddings/generate_embeddings.py roberta \
  --dataset bpRNA \
  --model-name roberta-base \
  --device cpu

python code/embeddings/generate_embeddings.py rnabert \
  --dataset bpRNA \
  --external-root external/RNABERT \
  --temp-dir /tmp/rnabert_embeddings
```

Common options:

- `--metadata-csv`: override `data/metadata/<dataset>.csv`
- `--output-root`: override the default embedding output directory
- `--ids-file`: process only a subset of sequence ids
- `--start-idx`: resume from a row offset
- `--limit`: process only the first `N` filtered sequences
- `--overwrite`: replace existing `.npy` files

Model-specific external dependencies:

- `rnafm`: `external/RNA-FM/` and its checkpoint
- `ernie`: `external/ERNIE-RNA/` and its checkpoint
- `rnabert`: `external/RNABERT/`, its checkpoint, and config
- `rinalmo`: `external/RiNALMo/` and its checkpoint
- `roberta`: `transformers` model download or local cache

Environment variables accepted by the backends:

- `RNAFM_REPO`, `RNAFM_CHECKPOINT`
- `ERNIE_RNA_REPO`, `ERNIE_RNA_CHECKPOINT`
- `RNABERT_REPO`, `RNABERT_CHECKPOINT`, `RNABERT_CONFIG`
- `RINALMO_REPO`, `RINALMO_CHECKPOINT`

Default relative locations expected by the embedding backends:

- `external/RNA-FM/RNA-FM_pretrained.pth`
- `external/ERNIE-RNA/checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt`
- `external/RNABERT/bert_mul_2.pth`
- `external/RNABERT/RNA_bert_config.json`
- `external/RiNALMo/weights/rinalmo_giga_pretrained.pt`

The locked upstream URLs and pinned commits are recorded in `external/model_sources.lock.json`.

The extraction code preserves the original dissertation conventions:

- RNA-FM saves transformer layers `1..12` as `layer_0..11` without trimming special tokens.
- ERNIE-RNA removes the leading and trailing special tokens before saving.
- RNABERT uses the original split-and-merge path for sequences of length `>= 440`.
- RiNALMo removes the outer special tokens from every saved layer.
- RoBERTa tokenizes the RNA sequence with spaces between nucleotides to recover one row per residue.
