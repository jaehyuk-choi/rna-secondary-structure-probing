# Experimental Workflow: RNA Secondary Structure Prediction with Language Model Probes

This document records the end-to-end experimental workflow used for the dissertation write-up.

**Base directory**: `dissertations/feb23/`

---

## 1. Overview

The pipeline studies whether embeddings from pretrained RNA language models contain recoverable base-pairing information and whether those signals improve downstream RNA secondary-structure prediction.

The main stages are:

1. Split the bpRNA dataset into `TR0`, `VL0`, `TS0`, and `new`.
2. Train structural probes over `(layer, k, seed)` settings and record validation loss and probe performance.
3. Select the best configuration on `VL0` only.
4. Run probe-only evaluation on `TS0` and `NEW`.
5. Generate base-pair bonus files from the selected probe outputs.
6. Sweep `α` in CPLfold under ViennaRNA and Contrafold backends.
7. Select the optimal `α` on `VL0` only and extract the corresponding `TS0` and `NEW` results.
8. Run downstream analyses such as length-bin summaries, `α`-F1 curves, and significance tests.

---

## 2. Data

### 2.1 bpRNA dataset

- **Paths**: `dissertations/data/bpRNA.csv`, `dissertations/data/bpRNA_splits.csv`
- **Columns**:
  - `bpRNA.csv`: `id`, `sequence`, `base_pairs`
  - `bpRNA_splits.csv`: `id`, `fold`, `partition`, `min_train_dist`
- **`base_pairs` format**: `[[i1, j1], [i2, j2], ...]` with 1-based indexing

### 2.2 Partitions

| Partition | Number of sequences | Role |
|-----------|---------------------|------|
| `TR0` | 10,814 | probe training |
| `VL0` | 1,300 | validation and model selection |
| `TS0` | 1,305 | in-distribution test |
| `new` | 5,401 | held-out test |

Selection is restricted to `VL0`. Neither `TS0` nor `NEW` is used to choose thresholds or `α`.

### 2.3 Models

The `feb23` setup includes six representation sources:

- `ernie`
- `roberta`
- `rnafm`
- `rinalmo`
- `onehot`
- `rnabert`

---

## 3. Structural probe training

### 3.1 Probe architecture

The main probe is `StructuralContactProbe`, which applies a low-rank projection:

\[
z_i = B^\top h_i \in \mathbb{R}^k, \quad
s_{ij} = z_i^\top z_j, \quad
p_{ij} = \sigma(s_{ij})
\]

- \(h_i \in \mathbb{R}^D\): embedding at nucleotide position \(i\)
- \(B \in \mathbb{R}^{D \times k}\): projection matrix
- \(p_{ij}\): predicted base-pair probability for positions \((i, j)\)

A full-rank bilinear probe is also present for comparison:

\[
s_{ij} = h_i^\top W h_j
\]

Implementation:
- `dissertations/jan22/models/bilinear_probe_model.py`

### 3.2 Training objective

For a sequence of length \(L\), ground-truth base pairs are converted to a contact map \(Y \in \{0,1\}^{L \times L}\). The model outputs logits \(S \in \mathbb{R}^{L \times L}\) and probabilities \(P = \sigma(S)\).

Training uses binary cross entropy:

\[
\mathcal{L}_{\text{BCE}}(S, Y)
= - \sum_{i<j} \left( Y_{ij}\log \sigma(S_{ij}) + (1-Y_{ij})\log (1-\sigma(S_{ij})) \right).
\]

Padding and variable-length regions are masked out during training.

### 3.3 Hyperparameters

- `layer`: embedding extraction layer (`0` to `33`)
- `k`: projection rank (`32`, `64`, `128`)
- `seed`: training seed
- `threshold`, `decoding_mode`: selected at decoding time on `VL0`, not optimized as training parameters

### 3.4 Threshold sweep and decoding mode

The threshold and decoding mode define how pairwise probabilities are converted into a discrete base-pair set. They are treated as validation-selected decoding parameters.

The decoding procedure is greedy max-one matching:

1. Start from candidate pairs \(C = \{(i,j): 1 \le i < j \le L\}\).
2. If `decoding_mode` is constrained, remove disallowed nucleotide pairs.
   - `canonical_constrained`: `AU`, `UA`, `CG`, `GC`
   - `canonical_wobble`: `AU`, `UA`, `CG`, `GC`, `GU`, `UG`
3. Repeatedly select the highest-scoring unused pair.
4. Stop when the best remaining score is at or below `τ`.
5. Mark both positions as used and continue.

Implementation:
- `dissertations/jan22/utils/evaluation.py:prob_to_pairs()`

The `VL0` threshold sweep uses:

\[
\tau \in \{0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95\}
\]

Typical outputs:
- `val_threshold_sweep.csv`
- `val_best_threshold.csv`

Example paths:
- `dissertations/jan22/results_updated/outputs/onehot/layer_0/k_32/seed_42/val_threshold_sweep.csv`
- `dissertations/jan22/results_updated/outputs/onehot/layer_0/k_32/seed_42/val_best_threshold.csv`

### 3.5 Best-config selection

Each model is selected independently using `VL0` only.

#### Step A: choose `(layer, k, seed, best_epoch)`

Training runs are compared by validation loss:

\[
e^* = \arg\min_e \mathcal{L}_{\text{val}}^{(e)}
\]

The corresponding summary table is:
- `dissertations/jan22/results_updated/summary/all_runs_summary.csv`

The current selection rule is:
- `min_val_loss`

#### Step B: choose `(decoding_mode, threshold)`

For the run selected in Step A:

\[
(\text{mode}^*, \tau^*) = \arg\max_{\text{mode}, \tau} \operatorname{F1}_{\text{VL0}}(\text{mode}, \tau)
\]

The final configuration is stored in:
- `dissertations/jan22/results_updated/summary/final_selected_config.csv`

Relevant fields include:
- `selected_layer`
- `selected_k`
- `selected_seed`
- `selected_best_epoch`
- `selected_best_threshold`
- `selected_decoding_mode`
- `selection_rule`

### 3.6 Probe-only evaluation on TS0 and NEW

After fixing the selected configuration, probe-only predictions are evaluated on `TS0` and `NEW` without CPLfold. This provides the direct contact-prediction baseline.

Reference implementation:
- `dissertations/feb8/scripts/evaluation/compute_feb8_probe_only_metrics.py`

### 3.7 Selected configuration summary

- **Selection rule**: `min_val_loss`
- **Config file**: `jan22/results_updated/summary/final_selected_config.csv`

| Model | Layer | k | Threshold | Decoding |
|-------|-------|---|-----------|----------|
| ernie | 11 | 32 | 0.95 | unconstrained |
| roberta | 11 | 32 | 0.90 | unconstrained |
| rnafm | 11 | 32 | 0.90 | unconstrained |
| rinalmo | 25 | 64 | 0.65 | unconstrained |
| onehot | 0 | 32 | 0.70 | canonical_constrained |
| rnabert | 0 | 128 | 0.65 | unconstrained |

---

## 4. Base-pair bonus generation

### 4.1 Algorithm

1. Load the trained probe checkpoint and projection matrix `B`.
2. Load nucleotide embeddings \(h \in \mathbb{R}^{L \times D}\) for each sequence.
3. Compute:

\[
z = h B \in \mathbb{R}^{L \times k}, \quad
p_{ij} = \sigma(z_i^\top z_j)
\]

4. Keep pairs with \(p_{ij} \ge \text{threshold}\) and `i < j`.
5. Save them as `base_pair_{model}_{seq_id}.txt`.

If the selected `decoding_mode` is canonical, the same mask is applied when generating bonus candidates so that CPLfold receives the same admissible pair set as probe-only decoding.

### 4.2 Output format

```text
i	j	score
1	335	0.961317
3	8	0.958443
...
```

- 1-based indexing
- tab-separated fields
- `score` is a probability in `[0, 1]`

### 4.3 Output paths

- `VL0`: `feb23/vl0_base_pairs/{model}/base_pair_{model}_{seq_id}.txt`
- `TS0`: `feb23/ts_base_pairs/{model}/base_pair_{model}_{seq_id}.txt`
- `new`: `feb23/new_base_pairs/{model}/base_pair_{model}_{seq_id}.txt`

### 4.4 Scripts

- `jan22/scripts/generation/generate_base_pairs.py` for `TS0` and `new`
- `feb23/scripts/generate_vl0_base_pairs.py` for `VL0`

---

## 5. CPLfold `α` sweep

### 5.1 CPLfold overview

- **Path**: `CPLfold_inter/CPLfold_inter.py`
- **Role**: adds base-pair bonus scores to the folding objective
- **Scoring rule**:

\[
\text{score} \mathrel{+}= \text{bonus}_{ij} \times \alpha_{\text{scaled}}
\]

- **Scaling**
  - ViennaRNA mode: \(\alpha_{\text{scaled}} = \alpha \times 100\)
  - Contrafold mode: \(\alpha_{\text{scaled}} = \alpha\)

### 5.2 Sweep range

- Range: `0.0` to `2.0`
- Step size: `0.02`
- Number of values: `101`
- `α=0`: no probe bonus

### 5.3 Backends

| Backend | Flag | Energy model |
|---------|------|--------------|
| ViennaRNA | `--V` | Turner 37 parameters |
| Contrafold | default | Contrafold parameters |

### 5.4 Execution pipeline

- **Script**: `feb23/scripts/run_split_pipeline.py`
- **Inputs**: partition (`VL0`, `TS0`, `new`), base-pair directory, model list
- **Process**: run CPLfold for each `(seq_id, model, α)` combination, generate a dot-bracket structure, then compare against ground truth

### 5.5 Outputs

- `detailed_results_{model}.csv`
- Columns:
  - `seq_id`
  - `model`
  - `alpha`
  - `threshold_used`
  - `f1`
  - `precision`
  - `recall`
  - `tp`
  - `fp`
  - `fn`
  - `predicted_count`
  - `energy`
  - `structure`

### 5.6 Result directories

| Partition | ViennaRNA | Contrafold |
|-----------|-----------|------------|
| `VL0` | `feb23/results_vl0/` | `feb23/results_vl0_contrafold/` |
| `TS0` | `feb23/results_ts0/` | `feb23/results_ts0_contrafold/` |
| `new` | `feb23/results_new/` | `feb23/results_new_contrafold/` |

---

## 6. Validation-based optimal `α`

### 6.1 Procedure

1. Run the full `α` sweep on `VL0`.
2. Select the `α` that maximizes mean `VL0` F1 for each model and backend.
3. Extract the corresponding `TS0` and `NEW` rows from existing `detailed_results_*.csv` files.

### 6.2 Script

- `feb8/scripts/validation_workflow/select_optimal_alpha_and_extract.py`
- For `feb23`, pass:
  - `--feb8-dir /projects/u6cg/jay/dissertations/feb23`
- Output:
  - `feb23/validation_based_optimal/val_optimal_results.csv`

Main columns:
- `Backend`
- `Model`
- `Val_opt_alpha`
- `Val_F1`
- `TS0_F1_at_opt`
- `NEW_F1_at_opt`

The `feb23` configuration includes `rnabert`, whereas the script defaults to five models, so `rnabert` support may require a local adjustment.

### 6.3 Leakage control

- `α` is selected on `VL0` only.
- `TS0` and `new` are never used for `α` selection.

---

## 7. Evaluation metrics

### 7.1 Definitions

- **TP**: predicted pair matches a ground-truth pair with shift tolerance
- **FP**: predicted pair is absent from ground truth
- **FN**: ground-truth pair is missed

\[
P = \frac{TP}{TP + FP}, \quad
R = \frac{TP}{TP + FN}, \quad
F1 = \frac{2PR}{P + R}
\]

### 7.2 Shift tolerance

- `shift=1`: a pair is counted as correct if it matches `(i±1, j)` or `(i, j±1)`
- Implementation:
  - `jan22/utils/evaluation.py`
  - `compute_pair_metrics`
  - `precision_recall_f1`

### 7.3 Evaluation flow

1. Convert CPLfold dot-bracket output into a predicted pair list.
2. Load bpRNA ground-truth pairs.
3. Compute TP, FP, and FN with shift tolerance.
4. Compute precision, recall, and F1.

---

## 8. Analysis

### 8.1 Length-bin analysis

- Bins: `<100 nt`, `100–200 nt`, `200–400 nt`, `≥400 nt`
- Metrics compared: `α=0.0`, fixed `α=1.0`, and validation-selected optimal `α`
- Statistics: paired t-tests with significance markers

### 8.2 `α`-F1 curves

- Plot mean F1, precision, and recall across `α`
- Stratify by model, partition, and backend

### 8.3 Scripts and outputs

- `feb8/scripts/analysis/generate_length_tables_and_graphs.py`
- Output directory:
  - `feb23/analysis_outputs/length_tables/`

---

## 9. End-to-end pipeline summary

```text
1. Prepare data
   bpRNA_splits.csv -> TR0, VL0, TS0, new

2. Train probe sweeps in jan22
   StructuralContactProbe over layer, k, seed

3. Select the best configuration on VL0
   (layer*, k*, epoch*) from validation loss
   (mode*, threshold*) from validation F1

4. Run probe-only evaluation on TS0 and NEW

5. Generate base-pair bonus files
   embeddings + B -> p_ij >= threshold -> base_pair_*.txt

6. Sweep CPLfold alpha
   alpha in [0.0, 0.02, ..., 2.0]

7. Select alpha on VL0 and extract TS0/NEW results

8. Run downstream analyses
```

---

## 10. Key paths

| Item | Path |
|------|------|
| Base directory | `dissertations/feb23/` |
| Data split file | `dissertations/data/bpRNA_splits.csv` |
| bpRNA sequences and structures | `dissertations/data/bpRNA.csv` |
| Embeddings | `dissertations/data/embeddings/{model}/bpRNA/by_layer/layer_{k}/` |
| Probe config | `dissertations/jan22/results_updated/summary/final_selected_config.csv` |
| Probe checkpoints | `dissertations/jan22/results_updated/outputs/{model}/layer_{k}/k_{d}/seed_42/best.pt` |
| VL0 base-pair generation | `feb23/scripts/generate_vl0_base_pairs.py` |
| CPLfold pipeline | `feb23/scripts/run_split_pipeline.py` |
| CPLfold executable | `CPLfold_inter/CPLfold_inter.py` |
| Optimal-`α` extraction | `feb8/scripts/validation_workflow/select_optimal_alpha_and_extract.py` |
| Evaluation utilities | `jan22/utils/evaluation.py` |
| Analysis outputs | `feb23/analysis_outputs/length_tables/` |

---

## 11. Example Slurm commands

```bash
cd /projects/u6cg/jay/dissertations/feb23

# VL0 base-pair generation
sbatch slurm/run_vl0_base_pairs.sh

# CPLfold α sweep (ViennaRNA)
sbatch slurm/run_vl0_ernie.sh
sbatch slurm/run_vl0_roberta.sh
sbatch slurm/run_vl0_rnafm.sh
sbatch slurm/run_vl0_rinalmo.sh
sbatch slurm/run_vl0_onehot.sh
sbatch slurm/run_vl0_rnabert.sh

# CPLfold α sweep (Contrafold)
sbatch slurm/run_vl0_ernie_contrafold.sh
sbatch slurm/run_vl0_roberta_contrafold.sh
# ... per model

# Extract validation-selected optimal α results
python3 /projects/u6cg/jay/dissertations/feb8/scripts/validation_workflow/select_optimal_alpha_and_extract.py \
  --feb8-dir /projects/u6cg/jay/dissertations/feb23
```
