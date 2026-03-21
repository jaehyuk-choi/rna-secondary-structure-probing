# REVIEW NOTES — Pre-Upload Checklist

Review these items before uploading to GitHub.

---

## CRITICAL: Missing Files Referenced by LaTeX

The following files are referenced by `\includegraphics` in `skeleton.tex` but were **not found anywhere** in the repository:

1. **`img/rna_structure.png`** (line 276)
   - Used in Section 2: RNA secondary structure illustration
   - This may be a manually created or externally sourced figure
   - **Action required:** Locate this file and add it to `figures/main/`

2. **`img/pipeline_figure.pdf`** (line 859)
   - Used in Section 3: High-level pipeline overview
   - This may be a manually created diagram (e.g., from TikZ, PowerPoint, or similar)
   - **Action required:** Locate this file and add it to `figures/main/`

3. **`img/max 1.pdf`** (line 1217)
   - Used in Section 3: Greedy max-one decoding illustration
   - **Action required:** Locate this file and add it to `figures/main/`

4. **`mybibfile.bib`** — bibliography file
   - Referenced by `\bibliography{mybibfile}` (line 3169)
   - Not found in the repository
   - **Action required:** Locate and add to `dissertation/`

5. **`infthesis.cls`** — dissertation document class
   - Referenced by `\documentclass[logo,bsc,singlespacing,parskip]{infthesis}` (line 5)
   - Standard Edinburgh Informatics thesis class file
   - **Action required:** Add to `dissertation/` (may be provided by the university)

6. **`ugcheck.sty`** — style file
   - Referenced by `\usepackage{ugcheck}` (line 6)
   - Standard Edinburgh UG check package
   - **Action required:** Add to `dissertation/` (may be provided by the university)

---

## LaTeX img/ Path

The LaTeX source uses `img/` as a relative path for figures (e.g., `\includegraphics{img/rna_structure.png}`). To compile the dissertation from this package, you will need to either:
- Create an `img/` symlink or directory in the same location as `skeleton.tex` and populate it with the figure files from `figures/main/` and `figures/appendix/`
- Or add `\graphicspath{{../figures/main/}{../figures/appendix/}}` to `skeleton.tex` and rename figures to match the `img/` prefix

---

## Environment Files

No environment specification files (requirements.txt, conda env YAML, etc.) were found in the repository. The `environment/` directory is currently empty.

**Action required:** Consider creating a `requirements.txt` or `environment.yml` with the key dependencies:
- Python version
- PyTorch
- NumPy, pandas, matplotlib, seaborn
- ViennaRNA Python bindings
- Any model-specific dependencies (transformers, etc.)

---

## Large Omitted Files

The following categories of files are omitted due to size but may be needed for full reproducibility:

1. **Embeddings** (`dissertations/data/embeddings/`) — frozen encoder embeddings
2. **Contact maps** (`dissertations/data/contact_maps/`) — binary .npy files
3. **Probe model checkpoints** (`.pt` files) — trained probe weights
4. **Per-sequence base pair files** (`feb25/base_pairs_thresholded/`) — CPLfold inputs

Consider sharing these via a data repository (e.g., Edinburgh DataShare, Zenodo) and linking from `data/README.md`.

---

## Items That Could Not Be Traced Confidently

1. **Table: probe_unconstrained_full numerical values** — The unconstrained results are in `results/metrics/unconstrained_results_summary.csv` (from march1). Verify that the CSV values exactly match the LaTeX table entries.

2. **Table: alpha_vl0_summary (Appendix)** — The VL0 alpha sweep table values should be checked against `results/sweeps/vl0_alpha_sweep_both.csv`.

3. **The appendix heatmap figure** — `figures/appendix/probe_contact_heatmaps.png` was taken from `march1/figures/probe_heatmaps_bprna_rfam_21498/`. Verify this is the correct version (for sequence 21498, not 22136 or the probe_heatmaps/ version).

---

## Verify Before Upload Checklist

- [ ] Locate and add `rna_structure.png`, `pipeline_figure.pdf`, `max 1.pdf`
- [ ] Locate and add `mybibfile.bib`
- [ ] Add `infthesis.cls` and `ugcheck.sty` (or note they're university-provided)
- [ ] Create environment specification file
- [ ] Verify all LaTeX table values match the CSV files in `results/`
- [ ] Test that `skeleton.tex` compiles with the provided figures
- [ ] Decide whether to share large data files externally
- [ ] Review `figures/appendix/probe_contact_heatmaps.png` is correct version
- [ ] Check if any additional LaTeX packages are needed
