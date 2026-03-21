# Data Directory

## Included Files

### metadata/
- **ArchiveII.csv** — ArchiveII RNA structure database metadata
- **bpRNA.csv** — bpRNA dataset metadata including sequence IDs, lengths, and family annotations

### splits/
- **bpRNA_splits.csv** — Partition assignments for all bpRNA sequences into:
  - **TR0** — training set
  - **VL0** — validation set (used for all hyperparameter selection)
  - **TS0** — in-distribution test set
  - **NEW** — out-of-distribution test set (unseen RNA families)

## Omitted (Too Large)

The following data files are required for full pipeline execution but are omitted from this package due to size:

### Contact Maps
- **Original location:** `dissertations/data/contact_maps/`
- **Format:** `.npy` binary files, one per sequence
- **Content:** Ground-truth binary contact matrices derived from bpRNA secondary structure annotations
- **Regeneration:** Run `code/preprocessing/compute_structure_features.py` with the bpRNA dataset

### Embeddings
- **Original location:** `dissertations/data/embeddings/`
- **Content:** Frozen encoder embeddings extracted from each model (ERNIE-RNA, RNA-FM, RoBERTa, RiNALMo, RNABERT) and one-hot encodings
- **Format:** Per-sequence embedding files
- **Regeneration:** Extract embeddings from each pretrained model using the sequences in bpRNA_splits.csv

### Per-Sequence Base Pair Files
- **Original location:** `dissertations/feb25/base_pairs_thresholded/`
- **Content:** Thresholded probe-derived base pair predictions for CPLfold input
- **Format:** Text files with (i, j, probability) triplets per sequence per model
- **Regeneration:** Run `code/probe_inference/generate_vl0_base_pairs.py` followed by `code/folding_integration/generate_thresholded_base_pairs.py`

### Probe Model Checkpoints
- **Original location:** `dissertations/feb23/models/` (and earlier iterations)
- **Content:** Trained probe weights (`.pt` files)
- **Regeneration:** Run `code/probe_training/run_split_pipeline.py`

## Expected Original Data Layout

The full pipeline expects data organised as follows:
```
data/
  bpRNA.csv                    # Metadata
  bpRNA_splits.csv             # Split assignments
  contact_maps/
    baseline1/                 # Contact maps per sequence
      {sequence_id}_contact.npy
  embeddings/
    {model_name}/              # Per-model embeddings
      {sequence_id}.npy
```

## Essential Split/Metadata Files

The following files in this package are essential for understanding and reproducing the experimental setup:
1. `splits/bpRNA_splits.csv` — defines which sequences go in TR0/VL0/TS0/NEW
2. `metadata/bpRNA.csv` — provides sequence metadata (length, family, etc.)
3. `metadata/ArchiveII.csv` — additional RNA structure annotations
