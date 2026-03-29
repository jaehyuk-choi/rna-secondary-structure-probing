# Integration Summary: Probe Training Pipeline

## What was added

The `code/probe_training/` directory now contains the minimum set of files
needed to understand and reproduce the bilinear probe training described in
the dissertation.  These files were copied from the upstream source tree
(`/projects/public/u5cj/dissertation/jan20/`) and adapted for the submission
layout.

### New files

| File | Role |
|---|---|
| `experiment_config.py` | Model registry, hyper-parameter grid, layer detection |
| `train_probe_automated.py` | Main training entry point (used by Slurm jobs) |
| `dataset_probe.py` | PyTorch Dataset for loading embeddings + contact maps |
| `utils_batch.py` | Padding/collation for variable-length RNA sequences |
| `sbatch_train_model.sh` | Slurm job script for one model (all layers x k values) |
| `run_all_experiments.sh` | Master script: submits train -> sweep -> summarise -> eval |
| `__init__.py` | Package marker |
| `SOURCE_MAP.md` | Old-path to new-path traceability table |
| This file | Integration rationale |

### Files already present (not duplicated)

- `code/models/bilinear_probe_model.py` — probe architecture (identical to upstream)
- `code/utils/evaluation.py` — metric computation (identical to upstream)

### Legacy inventory

- `docs/legacy_inventory/upstream_code_manifest.txt` — full file listing from
  the upstream source tree, for provenance.

## Why these files

The automated training pipeline (`train_probe_automated.py` invoked by
`sbatch_train_model.sh`) is the entry point that produced all reported
probe checkpoints.  Its direct dependencies are:
  `experiment_config` -> `dataset_probe` -> `utils_batch` (data loading),
  `bilinear_probe_model` (architecture), and `evaluation` (metrics).

Post-training scripts (threshold sweep, summarisation, final evaluation)
were excluded because they are downstream of training and not required for
a reader to understand or reproduce the probe training itself.

## What was excluded (large files)

| Category | Typical path | Reason |
|---|---|---|
| Trained checkpoints | `results/outputs/<model>/layer_*/k_*/seed_*/best.pt` | ~50 MB each; too large for submission |
| Precomputed embeddings | `data/embeddings/<MODEL>/bpRNA/by_layer/layer_*/*.npy` | Multi-GB per model |
| Contact maps | `data/contact_maps/bpRNA/*_contact.npy` | Derived from bpRNA annotations |
| Slurm logs | `results/slurm_logs/*.out`, `*.err` | Ephemeral |
| Generated figures | `img/` | Reproducible from plotting scripts |

These artefacts reside on the HPC filesystem and are referenced by path in
the training configuration (`experiment_config.ROOT`).

## Import changes

| Original import | Adapted import | Reason |
|---|---|---|
| `from utils.dataset_probe import ...` | `from dataset_probe import ...` | Module now lives alongside training script |
| `from .utils_batch import collate_rna_batch` | `from utils_batch import collate_rna_batch` | Absolute import; both modules in same directory on `sys.path` |
| `from experiment_config import ...` | *(unchanged)* | Script adds its own directory to `sys.path` |
| `from models.bilinear_probe_model import ...` | *(unchanged)* | Script adds `code/` to `sys.path` |
| `from utils.evaluation import ...` | *(unchanged)* | Resolved via `code/` on `sys.path` |

## Path portability

- `experiment_config.ROOT` defaults to `/projects/u5cj/jay/dissertation` but
  can be overridden with `export RNA_PROBE_DATA_ROOT=/your/path`.
- Shell scripts use `$PROJECT_ROOT` and `$CONDA_ENV` variables (with defaults).
