# Interpreting RNA Foundation Models via Structural Probing

**Jaehyuk Choi** — BSc Artificial Intelligence, University of Edinburgh (4th Year Project)

## Overview

This repository investigates whether pretrained RNA foundation models encode base-pairing structure in their learned representations. We train low-rank bilinear probes on frozen per-residue embeddings from five encoder-only RNA language models and a one-hot baseline, then evaluate whether the recovered contact information can improve thermodynamic structure prediction when injected as soft priors into CPLfold.

**Models tested:** ERNIE-RNA, RNA-FM, RoBERTa, RiNALMo, RNABERT, one-hot baseline
**Dataset:** bpRNA (TR0 / VL0 / TS0 / NEW splits)
**Key finding:** All five pretrained models recover significant base-pairing signal; ERNIE-RNA benefits most from CPLfold integration, while RoBERTa transfers best to unseen RNA families.

## Repository Structure

```
├── code/
│   ├── models/              Bilinear probe definition (StructuralContactProbe)
│   ├── embeddings/          Embedding extraction backends (6 models)
│   ├── probe_training/      Training pipeline + Slurm scripts
│   ├── probe_inference/     Generate base-pair scores from trained probes
│   ├── evaluation/          Metric computation, config selection, summary tables
│   ├── folding_integration/ CPLfold engine + alpha-sweep experiments
│   ├── analysis/            Canonical rates, statistical tests, pair statistics
│   ├── plotting/            All dissertation figure scripts
│   ├── preprocessing/       Contact map generation, dataset statistics
│   ├── utils/               Shared evaluation helpers
│   └── run_sh/              Shell launchers and Slurm job scripts
├── configs/                 Validation-selected hyperparameter configs
├── data/
│   ├── metadata/            bpRNA and ArchiveII sequence metadata
│   └── splits/              TR0/VL0/TS0/NEW partition assignments
├── results/
│   ├── metrics/             Probe-only F1, precision, recall (TS0/NEW)
│   ├── folding/             CPLfold alpha-sweep results
│   ├── sweeps/              Layer-wise and k-comparison validation sweeps
│   ├── per_sequence/        Per-sequence metrics for statistical tests
│   ├── tables/              Derived tables (canonical rates, pair distributions)
│   └── statistics/          Significance test outputs
├── figures/                 Generated dissertation figures (main + appendix)
├── external/                Upstream model repos and checkpoints (not committed)
├── dissertation/            LaTeX source (skeleton.tex)
├── scripts/                 External repo setup and validation helpers
└── requirements.txt         Python dependencies
```

## Key Results

### Probe-Only F1 (Unconstrained Decoding)

| Model | Layer | k | TS0 F1 | NEW F1 |
|-------|-------|---|--------|--------|
| ERNIE-RNA | 11 | 32 | 0.437 | 0.404 |
| RNA-FM | 11 | 32 | 0.139 | 0.103 |
| RoBERTa | 11 | 32 | 0.098 | 0.114 |
| RiNALMo | 10 | 32 | 0.016 | 0.024 |
| RNABERT | 0 | 128 | 0.026 | 0.026 |
| One-hot | 0 | 128 | 0.008 | 0.013 |

ERNIE-RNA achieves the highest probe-only F1 on both test sets. RoBERTa is the only model where NEW F1 exceeds TS0 F1, indicating stronger generalisation to unseen RNA families. RoBERTa significantly outperforms all models except ERNIE-RNA (Wilcoxon p < 0.001).

### CPLfold Integration (alpha=0 vs Best alpha)

| Model | TS0 Vienna (alpha=0 / best) | TS0 CONTRAfold (alpha=0 / best) | NEW CONTRAfold (alpha=0 / best) |
|-------|---------------------------|-------------------------------|-------------------------------|
| ERNIE-RNA | 0.544 / 0.656 (+0.112) | 0.560 / 0.683 (+0.123) | 0.630 / 0.738 (+0.108) |
| RNA-FM | 0.544 / 0.602 (+0.058) | 0.560 / 0.624 (+0.064) | 0.630 / 0.653 (+0.023) |
| RoBERTa | 0.544 / 0.555 (+0.011) | 0.560 / 0.587 (+0.027) | 0.630 / 0.660 (+0.030) |

Injecting probe scores as soft priors into CPLfold consistently improves structure prediction. ERNIE-RNA yields the largest absolute gain (+12.3% F1 on TS0 CONTRAfold).

## Probe Architecture

The structural probe applies a low-rank bilinear projection to frozen per-residue embeddings:

```
z_i = B^T h_i  ∈ R^k
s_ij = z_i^T z_j
p_ij = σ(s_ij)
```

where `h_i ∈ R^D` is the embedding at position `i`, `B ∈ R^{D×k}` is the learned projection matrix, and `p_ij` is the predicted base-pair probability for positions `(i, j)`. Training uses binary cross-entropy against ground-truth contact maps. Decoding uses greedy max-one matching: pairs are selected in descending probability order, each position pairs at most once, and a threshold `τ` is applied. This does not enforce pseudoknot-freeness.

Implementation: `code/models/bilinear_probe_model.py`

## Task-to-Code Mapping

| Task | Code | Key inputs | Key outputs |
|------|------|-----------|-------------|
| **Probe model** | `code/models/bilinear_probe_model.py` | — | `StructuralContactProbe` class |
| **Embedding extraction** | `code/embeddings/generate_embeddings.py` | `data/metadata/*.csv`, external model repos | `data/embeddings/{MODEL}/{DATASET}/by_layer/layer_*/` |
| **Training** | `code/probe_training/train_probe_automated.py` | Embeddings, contact maps | `results/outputs/{model}/layer_*/k_*/seed_*/best.pt` |
| **Slurm training** | `code/probe_training/sbatch_train_model.sh` | Model name | All (layer, k) checkpoints for one model |
| **Full pipeline** | `code/probe_training/run_all_experiments.sh` | — | All training + downstream jobs |
| **Config selection** | `code/evaluation/select_unconstrained_best_config.py` | `results/outputs/` | `configs/final_selected_config_unconstrained.csv` |
| **TS0/NEW metrics** | `code/evaluation/compute_probe_only_metrics.py` | Configs, checkpoints | `results/metrics/final_{test,new}_metrics.csv` |
| **Wobble metrics** | `code/evaluation/compute_probe_only_with_wobble.py` | Configs, checkpoints | `results/metrics/*_wobble.csv` |
| **Summary table** | `code/evaluation/build_summary_table.py` | Final metrics | `results/metrics/unconstrained_results_summary.csv` |
| **CPLfold engine** | `code/folding_integration/CPLfold_inter.py` | Sequence, energy params, bonus matrix | Dot-bracket structure |
| **CPLfold experiments** | `code/folding_integration/run_cplfold_exp.py` | Base pairs, configs | `results/folding/detailed_alpha_sweep_*.csv` |
| **VL0 alpha sweep** | `code/probe_training/run_split_pipeline.py` | VL0 sequences, probe scores | `results/sweeps/vl0_alpha_sweep_both.csv` |
| **Canonical rates** | `code/analysis/build_canonical_rate_table_with_baseline.py` | Wobble metrics, baseline | `results/tables/canonical_rate_table_with_baseline.csv` |
| **RoBERTa stats** | `code/analysis/roberta_vs_others_statistical_test.py` | Per-sequence metrics | `results/statistics/roberta_vs_others_significance.csv` |
| **Pair statistics** | `code/analysis/compute_pair_statistics.py` | bpRNA metadata, splits | `results/tables/pair_statistics_by_split.csv` |
| **Figures** | `code/plotting/plot_*.py` | Various results CSVs | `figures/main/*.png`, `figures/appendix/*.png` |

## Dissertation Results Mapping

| Dissertation reference | Result file |
|----------------------|-------------|
| Table 5 — Probe-only F1 (unconstrained) | `results/metrics/unconstrained_results_summary.csv` |
| Table 6 — Canonical pairing rates | `results/tables/canonical_rate_table_with_baseline.csv` |
| Table 7 — Pair type distributions | `results/tables/{onehot,rinalmo}_pair_combo_distribution.csv` |
| Table 8 — Selected hyperparameters | `configs/final_selected_config_unconstrained.csv` |
| Table 9 — Probe-only F1 (best config) | `results/metrics/final_{test,new}_metrics.csv` |
| Table 11 — Unconstrained vs best comparison | `results/metrics/probe_unconstrained_vs_best_comparison.csv` |
| Table 12 — F1 by sequence length | `results/per_sequence/{ts,new}_per_sequence_metrics.csv` |
| Table 13 — RoBERTa significance tests | `results/statistics/roberta_vs_others_significance.csv` |
| Fig — α sweep (VL0) | `results/sweeps/vl0_alpha_sweep_both.csv` |
| Fig — α=0 vs best α | `results/folding/alpha0_vs_best_full.csv` |
| Fig — Layer-wise F1 | `results/sweeps/layer_wise_val_f1.csv` |
| Fig — k comparison | `results/sweeps/k_comparison_val_f1.csv` |
| CPLfold optimal α | `configs/val_optimal_results.csv` |

## Reproduction

### Prerequisites

- Python 3.10+, PyTorch 1.13+ (CUDA recommended)
- `pip install -r requirements.txt`
- Optional: ViennaRNA Python bindings (`conda install -c bioconda viennarna`)

### What is included vs excluded

This repository contains all source code, configuration files, pre-computed result CSVs, and generated figures needed to understand and verify the dissertation results. The following large artifacts are **not included** due to size:

| Artefact | Expected location | How to regenerate |
|----------|------------------|-------------------|
| Per-layer embeddings | `data/embeddings/{MODEL}/bpRNA/by_layer/layer_{N}/{id}.npy` | See [code/embeddings/README.md](code/embeddings/README.md) |
| Contact maps | `data/contact_maps/bpRNA/{id}_contact.npy` | `python code/preprocessing/compute_structure_features.py` |
| Trained checkpoints | `results/outputs/{model}/layer_*/k_*/seed_*/best.pt` | `bash code/probe_training/run_all_experiments.sh` |
| Per-model CPLfold detailed results | `results/folding/{subdir}/detailed_results_{model}.csv` | `python code/folding_integration/run_cplfold_exp.py` |

The pre-computed result CSVs in `results/` are sufficient to verify all dissertation tables and regenerate all figures without rerunning experiments.

### External model dependencies

Embedding generation requires upstream model repositories and pretrained checkpoints. These are cloned under `external/` and pinned to specific commits via `external/model_sources.lock.json`.

```bash
bash scripts/setup_external_repos.sh
bash scripts/check_external_assets.sh
```

| Model | Upstream repository | Checkpoint |
|-------|-------------------|------------|
| RNA-FM | [ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM) | `RNA-FM_pretrained.pth` |
| ERNIE-RNA | [Bruce-ywj/ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA) | `ERNIE-RNA_pretrain.pt` |
| RNABERT | [mana438/RNABERT](https://github.com/mana438/RNABERT) | `bert_mul_2.pth` |
| RiNALMo | [lbcb-sci/RiNALMo](https://github.com/lbcb-sci/RiNALMo) | `rinalmo_giga_pretrained.pt` |
| RoBERTa | Hugging Face (`roberta-base`) | Downloaded automatically |
| One-hot | — | — |

See [code/embeddings/README.md](code/embeddings/README.md) for detailed model-by-model extraction instructions.

### Generate embeddings

```bash
python code/embeddings/generate_embeddings.py onehot --dataset bpRNA
python code/embeddings/generate_embeddings.py roberta --dataset bpRNA --device cpu
python code/embeddings/generate_embeddings.py rnafm --dataset bpRNA --device cpu
python code/embeddings/generate_embeddings.py ernie --dataset bpRNA --device cpu
python code/embeddings/generate_embeddings.py rnabert --dataset bpRNA --temp-dir /tmp/rnabert_embeddings
python code/embeddings/generate_embeddings.py rinalmo --dataset bpRNA --device cuda
```

### Train a single probe

```bash
python code/probe_training/train_probe_automated.py \
    --model rnafm --layer 11 --k 64 --seed 42
```

### Run full Slurm pipeline (all models)

```bash
bash code/probe_training/run_all_experiments.sh
```

### Compute probe-only metrics

```bash
bash code/run_sh/run_probe_only.sh
```

### Generate figures

```bash
bash code/run_sh/run_all_plots.sh
```

## Experimental Parameters

| Parameter | Values |
|-----------|--------|
| Models | ERNIE-RNA, RNA-FM, RoBERTa, RiNALMo, RNABERT, one-hot |
| Projection rank (k) | 32, 64, 128 |
| Decoding modes | unconstrained, canonical-only, canonical+wobble |
| Threshold (τ) | Swept 0.50–0.95 on VL0 |
| CPLfold α | Swept 0.0–2.0 (step 0.02) on VL0 |
| Folding backends | ViennaRNA, CONTRAfold |
| Splits | TR0 (train), VL0 (val), TS0 (in-distribution test), NEW (OOD test) |

## Reproducibility Notes

**Fully reproducible from this repository alone:** All dissertation tables and figures can be verified from the pre-computed CSVs in `results/`. Most plotting scripts read directly from these CSVs and can regenerate figures without any external data.

**Reproducible with external data:** The probe training and evaluation pipeline is fully wired from dataset metadata through embedding generation, probe training, and probe-only evaluation. Given the external model checkpoints and sufficient compute, the full training pipeline can be rerun.

**CPLfold integration:** The CPLfold folding engine (`code/folding_integration/CPLfold_inter.py`) and the experiment orchestration scripts are included and logically complete. However, the CPLfold downstream analysis scripts (`compare_alpha0_vs_best.py`, `alpha0_vs_best_stats.py`, `aggregate_cplfold_val_ts0_new.py`) expect per-model detailed result directories that are not bundled due to size. The aggregated results in `results/folding/` and `results/sweeps/` are included and support all dissertation claims.

**LaTeX compilation:** The dissertation source (`dissertation/skeleton.tex`) references figures via `img/` relative paths and requires `infthesis.cls`, `ugcheck.sty`, and `mybibfile.bib`, which are standard University of Edinburgh thesis infrastructure files not included in this repository.

## License

This repository is submitted as part of a BSc dissertation at the University of Edinburgh.
