# Embedding Generation Pipeline

This document describes the current embedding-generation pipeline in this repository, including:

- what files are used as inputs
- what external repositories and checkpoints are required
- how each model is executed
- what files are written to disk
- how the generated embeddings connect to the downstream probe-training code

The description here matches the code currently stored under `code/embeddings/`.

## 1. Purpose

The embedding pipeline converts RNA sequences from the repository metadata tables into per-sequence, per-layer `.npy` files.

These files are the inputs to the structural probing experiments. Downstream training and evaluation code expects them under:

```text
data/embeddings/<MODEL>/<DATASET>/by_layer/layer_<N>/<seq_id>.npy
```

Examples:

```text
data/embeddings/Onehot/bpRNA/by_layer/layer_0/bpRNA_RFAM_37268.npy
data/embeddings/RNAFM/bpRNA/by_layer/layer_11/bpRNA_RFAM_37268.npy
data/embeddings/RNABERT/bpRNA/by_layer/layer_5/bpRNA_RFAM_37268.npy
```

## 2. Code Entry Points

Main CLI:

```text
code/embeddings/generate_embeddings.py
```

Common path and metadata helpers:

```text
code/embeddings/common.py
```

Model-specific backends:

```text
code/embeddings/backends/onehot_backend.py
code/embeddings/backends/roberta_backend.py
code/embeddings/backends/rnafm_backend.py
code/embeddings/backends/ernie_rna_backend.py
code/embeddings/backends/rnabert_backend.py
code/embeddings/backends/rinalmo_backend.py
```

External setup and validation:

```text
scripts/setup_external_repos.sh
scripts/check_external_assets.sh
external/model_sources.lock.json
external/README.md
```

## 3. Input Files Inside This Repository

### 3.1 Metadata tables

The pipeline reads sequence metadata from:

```text
data/metadata/bpRNA.csv
data/metadata/ArchiveII.csv
```

The required columns are:

- `id`
- `sequence`

If a table uses `seq` instead of `sequence`, the loader renames it automatically.

### 3.2 Output root

By default, embeddings are written under:

```text
data/embeddings/
```

The exact model directory names are:

- `onehot` -> `Onehot`
- `roberta` -> `RoBERTa`
- `rnafm` -> `RNAFM`
- `ernie` -> `RNAErnie`
- `rnabert` -> `RNABERT`
- `rinalmo` -> `RiNALMo`

### 3.3 Existing downstream expectation

Probe training later reads embeddings from:

```text
data/embeddings/<MODEL_DIR>/bpRNA/by_layer/layer_<N>/
```

The current training path resolution is tied to `code/probe_training/experiment_config.py` and `code/probe_training/train_probe_automated.py`.

## 4. External Repositories and Checkpoints

The pretrained foundation-model code is not committed into this repository. Instead, it is expected under:

```text
external/
```

The setup helper clones the upstream repositories and checks out pinned commits:

```bash
bash scripts/setup_external_repos.sh
```

The validation helper reports any missing repositories, configs, or checkpoints:

```bash
bash scripts/check_external_assets.sh
```

The locked upstream URLs and pinned commits are recorded in:

```text
external/model_sources.lock.json
```

### 4.1 Expected external layout

```text
external/
├── RNA-FM/
│   └── RNA-FM_pretrained.pth
├── ERNIE-RNA/
│   ├── checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt
│   └── src/dict/
├── RNABERT/
│   ├── MLM_SFP.py
│   ├── RNA_bert_config.json
│   └── bert_mul_2.pth
└── RiNALMo/
    └── weights/rinalmo_giga_pretrained.pt
```

### 4.2 What setup does and does not do

`scripts/setup_external_repos.sh`:

- clones the four upstream repositories
- checks out pinned commits
- creates a few empty checkpoint directories when useful

It does not download every pretrained weight automatically. Some models still require manual checkpoint placement following upstream instructions.

## 5. End-to-End Flow

The embedding-generation process is:

1. Prepare upstream repositories under `external/`
2. Place any missing pretrained checkpoints in the expected paths
3. Validate external assets
4. Run the embedding CLI for one model at a time
5. Write per-sequence `.npy` files under `data/embeddings/...`
6. Reuse those embeddings in probe training and evaluation

In shell form:

```bash
cd <REPO_ROOT>

bash scripts/setup_external_repos.sh
bash scripts/check_external_assets.sh

python code/embeddings/generate_embeddings.py onehot --dataset bpRNA
python code/embeddings/generate_embeddings.py roberta --dataset bpRNA --device cpu
python code/embeddings/generate_embeddings.py rnafm --dataset bpRNA --device cpu
python code/embeddings/generate_embeddings.py ernie --dataset bpRNA --device cpu
python code/embeddings/generate_embeddings.py rnabert --dataset bpRNA --temp-dir /tmp/rnabert_embeddings
python code/embeddings/generate_embeddings.py rinalmo --dataset bpRNA --device cuda
```

## 6. Common CLI Behavior

All model subcommands support the same general controls:

- `--dataset`: `bpRNA` or `ArchiveII`
- `--metadata-csv`: override the metadata table path
- `--output-root`: override the embedding output root
- `--ids-file`: restrict extraction to a list of sequence ids
- `--start-idx`: resume from a row offset in the filtered metadata table
- `--limit`: process only the first `N` filtered sequences
- `--overwrite`: regenerate files even if they already exist

This means the same pipeline can be used for:

- full extraction on `bpRNA`
- partial reruns
- resuming long jobs
- debugging on a small subset

### 6.1 Skip behavior

For each sequence, the code checks whether all expected `layer_<N>/<seq_id>.npy` files already exist.

- if they all exist and `--overwrite` is not set, the sequence is skipped
- if any expected layer file is missing, the sequence is recomputed

## 7. Model-by-Model Details

### 7.1 One-hot

Code:

```text
code/embeddings/backends/onehot_backend.py
```

Input requirements:

- no external repository
- no pretrained checkpoint

Procedure:

1. Read `sequence` from the metadata table
2. Convert each nucleotide to a 4-dimensional one-hot vector
3. Use a uniform vector `[0.25, 0.25, 0.25, 0.25]` for unknown symbols
4. Save one file per sequence under `layer_0`

Output:

- one layer only
- shape `(L, 4)`

Output path pattern:

```text
data/embeddings/Onehot/<DATASET>/by_layer/layer_0/<seq_id>.npy
```

### 7.2 RoBERTa

Code:

```text
code/embeddings/backends/roberta_backend.py
```

Input requirements:

- no `external/` repository
- `transformers` installation
- ability to load the specified Hugging Face model, default `roberta-base`

Procedure:

1. Load `RobertaTokenizer` and `RobertaModel`
2. Convert each RNA sequence into a space-separated form such as `"A C G U"`
3. Tokenize with special tokens enabled
4. Run the transformer with `output_hidden_states=True`
5. Discard the embedding layer and keep transformer hidden states only
6. Reconstruct a residue-aligned `(L, D)` matrix by walking over tokens and skipping special markers such as `<s>`, `</s>`, `</w>`
7. Save one file per layer per sequence

Important behavior:

- token-to-residue alignment is reconstructed manually
- if the tokenizer output is shorter than the original sequence length, the last vector is repeated
- if it is longer, it is truncated

Output:

- typically 12 layers for `roberta-base`
- shape `(L, hidden_size)` for each layer

Output path pattern:

```text
data/embeddings/RoBERTa/<DATASET>/by_layer/layer_<0..11>/<seq_id>.npy
```

### 7.3 RNA-FM

Code:

```text
code/embeddings/backends/rnafm_backend.py
```

Input requirements:

- `external/RNA-FM/`
- `external/RNA-FM/RNA-FM_pretrained.pth`

Procedure:

1. Verify that the upstream repository and checkpoint exist
2. Import `rna_fm_t12` from the upstream `fm.pretrained` package
3. Load the pretrained model
4. Convert each sequence to tokens using the upstream alphabet batch converter
5. Request `repr_layers=0..12`
6. Ignore representation `0` and save transformer layers `1..12`
7. Store upstream layer `1` as local `layer_0`, upstream layer `12` as local `layer_11`

Important behavior:

- current code does not trim special tokens for RNA-FM
- it writes exactly what comes out of `representations[layer_idx + 1]`

Output:

- 12 saved layers
- usually shape `(token_length, D)` where token length may reflect the upstream tokenization convention

Output path pattern:

```text
data/embeddings/RNAFM/<DATASET>/by_layer/layer_<0..11>/<seq_id>.npy
```

### 7.4 ERNIE-RNA

Code:

```text
code/embeddings/backends/ernie_rna_backend.py
```

Input requirements:

- `external/ERNIE-RNA/`
- `external/ERNIE-RNA/checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt`
- `external/ERNIE-RNA/src/dict/`

Procedure:

1. Verify that the upstream repository, checkpoint, and token dictionary are present
2. Import `extract_embedding` from the upstream repository
3. For each sequence, call `extract_embedding_of_ernierna`
4. Expect a tensor shaped roughly like `(1, 12, L+2, 768)`
5. Remove the batch axis
6. Remove the first and last token positions with `[:, 1:-1, :]`
7. Save each of the 12 layers as a separate `.npy`

Important behavior:

- leading and trailing special tokens are removed before saving

Output:

- 12 saved layers
- shape `(L, 768)` per layer in the common case

Output path pattern:

```text
data/embeddings/RNAErnie/<DATASET>/by_layer/layer_<0..11>/<seq_id>.npy
```

### 7.5 RNABERT

Code:

```text
code/embeddings/backends/rnabert_backend.py
```

Input requirements:

- `external/RNABERT/`
- `external/RNABERT/MLM_SFP.py`
- `external/RNABERT/bert_mul_2.pth`
- `external/RNABERT/RNA_bert_config.json`

Procedure:

1. Verify that the upstream repository, entry script, checkpoint, and config are present
2. Load the upstream config JSON to determine the number of layers
3. Normalize ambiguity codes into `A/C/G/U`
4. For each sequence, write a temporary FASTA file
5. Call upstream `MLM_SFP.py` as a subprocess
6. Parse the emitted text file into a numeric array
7. Validate that the parsed array has shape `(num_hidden_layers, seq_len, 120)`
8. Save each layer to `layer_<N>/<seq_id>.npy`

Long-sequence handling:

- if sequence length is below `440`, RNABERT runs once
- if sequence length is `>= 440`, the sequence is split into two overlapping fragments
- both fragments are processed independently
- the two embedding blocks are merged with the original split-and-merge rule used in the older codebase

Error handling:

- failures can be written to a CSV with `--error-log`

Output:

- number of layers comes from `RNA_bert_config.json`
- expected hidden size is `120`

Output path pattern:

```text
data/embeddings/RNABERT/<DATASET>/by_layer/layer_<N>/<seq_id>.npy
```

### 7.6 RiNALMo

Code:

```text
code/embeddings/backends/rinalmo_backend.py
```

Input requirements:

- `external/RiNALMo/`
- `external/RiNALMo/weights/rinalmo_giga_pretrained.pt`

Procedure:

1. Verify that the upstream repository and checkpoint are present
2. Insert the external repository into `sys.path`
3. Install a fallback stub if `flash_attn` is unavailable
4. Dynamically load the upstream `rinalmo` modules
5. Construct the `giga` model configuration
6. Load the checkpoint state dict
7. Wrap the model with forward hooks to collect each block output
8. Tokenize each sequence with the upstream alphabet
9. Run a forward pass
10. Collect:
    - the embedding-layer output
    - intermediate transformer block outputs
    - the final representation
11. Remove batch dimension if present
12. Remove outer special tokens from every saved layer
13. Write each layer to disk

Important behavior:

- the number of saved layers depends on the upstream block count plus the embedding layer
- `flash_attn` is optional in this pipeline; if not available, the fallback path is used

Output path pattern:

```text
data/embeddings/RiNALMo/<DATASET>/by_layer/layer_<N>/<seq_id>.npy
```

## 8. Output File Semantics

Every saved file is a NumPy array with one row per saved token or residue position.

At a high level:

- first dimension: sequence length or token-aligned length
- second dimension: hidden dimension for the model

The exact row count differs slightly by model because token handling differs:

- `onehot`: always residue-aligned
- `roberta`: residue-aligned after manual reconstruction
- `ernie`: residue-aligned after trimming the outer special tokens
- `rnabert`: residue-aligned according to the upstream script output
- `rinalmo`: outer special tokens removed
- `rnafm`: current code keeps the representation returned by upstream transformer layers without extra trimming

## 9. Typical Commands

Prepare external repositories:

```bash
bash scripts/setup_external_repos.sh
bash scripts/check_external_assets.sh
```

Generate one-hot embeddings for the full bpRNA table:

```bash
python code/embeddings/generate_embeddings.py onehot --dataset bpRNA
```

Generate RoBERTa embeddings:

```bash
python code/embeddings/generate_embeddings.py roberta \
  --dataset bpRNA \
  --model-name roberta-base \
  --device cpu
```

Generate RNA-FM embeddings:

```bash
python code/embeddings/generate_embeddings.py rnafm \
  --dataset bpRNA \
  --device cpu
```

Generate ERNIE-RNA embeddings:

```bash
python code/embeddings/generate_embeddings.py ernie \
  --dataset bpRNA \
  --device cpu
```

Generate RNABERT embeddings:

```bash
python code/embeddings/generate_embeddings.py rnabert \
  --dataset bpRNA \
  --temp-dir /tmp/rnabert_embeddings \
  --error-log results/rnabert_embedding_errors.csv
```

Generate RiNALMo embeddings:

```bash
python code/embeddings/generate_embeddings.py rinalmo \
  --dataset bpRNA \
  --device cuda
```

Run only on a subset of sequences:

```bash
python code/embeddings/generate_embeddings.py rnafm \
  --dataset bpRNA \
  --ids-file path/to/ids.txt \
  --device cpu
```

Resume from a later row:

```bash
python code/embeddings/generate_embeddings.py rnabert \
  --dataset bpRNA \
  --start-idx 10000
```

## 10. Failure Modes and Checks

### 10.1 Missing external assets

If a required repository, checkpoint, config, or token dictionary is missing, the backend raises `FileNotFoundError` early.

Use:

```bash
bash scripts/check_external_assets.sh
```

before launching long jobs.

### 10.2 Missing Python packages

Common package requirements include:

- `numpy`
- `pandas`
- `tqdm`
- `torch`
- `transformers` for RoBERTa

Model-specific upstream repositories may have additional dependency requirements beyond this repository’s base `requirements.txt`.

### 10.3 RNABERT subprocess failures

RNABERT is the most brittle backend because it shells out to an upstream script and parses a text result. Use `--error-log` to capture failures cleanly.

### 10.4 RiNALMo attention dependency

If `flash_attn` is not installed, the backend falls back to a stubbed path and disables flash attention in the loaded config.

## 11. Where Embeddings Go Next

The generated embedding files feed directly into structural probe training.

Relevant downstream files:

```text
code/probe_training/experiment_config.py
code/probe_training/train_probe_automated.py
code/probe_training/dataset_probe.py
```

Training expects the embedding layout already described above. Once the `.npy` files exist, the rest of the probe pipeline can run without modification.

Example downstream step:

```bash
python code/probe_training/train_probe_automated.py \
  --model rnafm --layer 11 --k 64 --seed 42
```

## 12. Minimal Reproducibility Checklist

- `data/metadata/bpRNA.csv` exists
- external repositories are cloned under `external/`
- required checkpoints and config files are present
- `bash scripts/check_external_assets.sh` passes
- the chosen embedding backend finishes and writes `data/embeddings/...`
- probe training sees the expected layer directories under `data/embeddings/.../by_layer/`

## 13. Quick Reference Table

| Model | External repo needed | Checkpoint needed | Saved layers | Special-token handling |
|------|-----------------------|-------------------|--------------|------------------------|
| `onehot` | No | No | `layer_0` | Not applicable |
| `roberta` | No local repo | Hugging Face model/cache | Transformer layers only | Reconstructed to residue level |
| `rnafm` | `external/RNA-FM` | `RNA-FM_pretrained.pth` | 12 | No extra trimming in current code |
| `ernie` | `external/ERNIE-RNA` | `ERNIE-RNA_pretrain.pt` | 12 | Trim first and last positions |
| `rnabert` | `external/RNABERT` | `bert_mul_2.pth` + config | Config-dependent | Follows upstream script output |
| `rinalmo` | `external/RiNALMo` | `rinalmo_giga_pretrained.pt` | Block-count dependent | Trim first and last positions |

