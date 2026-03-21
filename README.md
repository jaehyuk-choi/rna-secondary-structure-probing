# Interpreting RNA Foundation Models: Evidence of Structural Awareness in Pretrained Representations

**Author:** Jaehyuk Choi
**Degree:** BSc Artificial Intelligence, University of Edinburgh
**Type:** 4th Year Project Report

## Project Summary

This dissertation investigates whether pretrained RNA foundation models (RNA FMs) encode base-pairing information in their representations, and whether such information can improve thermodynamic structure prediction. Using low-rank bilinear probes on frozen embeddings from five encoder-only models (ERNIE-RNA, RNA-FM, RoBERTa, RiNALMo, RNABERT) and a one-hot baseline, we assess decodability under varying pairing constraints, then inject probe-derived scores as soft priors into CPLfold under ViennaRNA and CONTRAfold backends.

## About This Package

This is a **final, purpose-organised dissertation submission package**. It contains only the canonical files that were actually used to produce the final dissertation and its reported results. It is organised by functional purpose, not by development chronology.

This package was rebuilt from scratch from the original repository contents, guided by the dissertation LaTeX source (`skeleton.tex`) as the primary source of truth.

## Package Structure

| Directory | Contents |
|---|---|
| `dissertation/` | LaTeX source and workflow documentation |
| `code/models/` | Bilinear probe model definition |
| `code/preprocessing/` | Dataset preparation and feature computation |
| `code/probe_training/` | Probe training pipeline |
| `code/probe_inference/` | Base pair score generation from trained probes |
| `code/evaluation/` | Metric computation and configuration selection |
| `code/folding_integration/` | CPLfold integration (ViennaRNA/CONTRAfold) |
| `code/analysis/` | Canonical rate, pair distribution, statistical tests |
| `code/plotting/` | All dissertation figure generation scripts |
| `code/utils/` | Shared evaluation utilities |
| `code/run_sh/` | Shell launcher scripts |
| `configs/` | Validation-selected model configurations |
| `results/` | Final metrics, per-sequence outputs, tables, sweeps, folding |
| `figures/` | Dissertation-referenced figure images (main + appendix) |
| `data/` | Dataset metadata and split assignments |
| `docs/` | Technical documentation |
| `environment/` | Environment specifications (see REVIEW_NOTES.md) |
| `legacy_optional/` | Reserved for provenance-only files (currently empty) |

## Key Entry Points

- **Dissertation source:** `dissertation/skeleton.tex`
- **Canonical code:** `code/` (organised by pipeline stage)
- **Canonical results:** `results/` (organised by type)
- **Figures:** `figures/main/` (9 main figures) + `figures/appendix/` (1 appendix figure)

## Intentional Exclusions

This package intentionally excludes:
- All date-based folder organisation (jan22, feb8, feb23, feb25, march1)
- Duplicate/superseded script iterations
- Cosmetic figure variants (lightorange, PDF duplicates)
- Intermediate result files not tied to dissertation tables/figures
- ~90 Slurm launch scripts that differ only by model/backend
- Checkpoint weights, embeddings, contact maps (too large)
- Debug scripts, logs, `__pycache__`, slurm output files

See `EXCLUDED_OR_SUPERSEDED.md` for full details.

## High-Level Pipeline

1. **Preprocessing:** Compute contact maps from bpRNA annotations (`code/preprocessing/`)
2. **Probe training:** Train bilinear probes on frozen embeddings (`code/probe_training/`)
3. **Probe inference:** Generate pairwise base-pair scores (`code/probe_inference/`)
4. **Evaluation:** Compute F1/precision/recall, select configs on VL0 (`code/evaluation/`)
5. **Folding integration:** Inject probe scores into CPLfold (`code/folding_integration/`)
6. **Analysis:** Canonical rates, statistical tests, length analysis (`code/analysis/`)
7. **Plotting:** Generate all dissertation figures (`code/plotting/`)

## Documentation Index

| File | Purpose |
|---|---|
| `MANIFEST.md` | Detailed inventory of included files |
| `RESULTS_TRACEABILITY.md` | Maps dissertation tables/figures to supporting files |
| `FILE_SELECTION_DECISIONS.md` | Explains deduplication and selection decisions |
| `EXCLUDED_OR_SUPERSEDED.md` | Lists intentionally excluded files with reasons |
| `REVIEW_NOTES.md` | Pre-upload checklist and unresolved items |
| `TREE.txt` | Readable directory tree |
