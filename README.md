# Grokked Transformer Quantization

**Research artifact for:**  
**Train–Test Selectivity and Architectural Fragility Under Quantization of Grokked Transformers**

**Authors:** Md Muntaqim Meherab and Mahathir Muhammad  
**Affiliation:** Daffodil International University, Dhaka, Bangladesh

This repository contains the executed experiment notebooks, archived datasets, consolidated numerical exports, publication-facing figures, notebook-generated tables, and reproducibility documentation for our study of post-training quantization (PTQ) in grokked transformers.

> **Manuscript status.** This repository accompanies a research manuscript. It does not imply acceptance or publication by any journal or conference.

## Research question

We ask whether untargeted post-training weight quantization preferentially damages the generalization acquired during grokking, or whether training and test performance mainly degrade together. The experiments use one-layer SwiGLU transformers trained on modular addition, multiplication, and subtraction at prime moduli \(p\in\{59,97\}\).

The primary selectivity estimand compares each quantized checkpoint with its own full-precision baseline:

\[
\Delta G(b)=100[\mathrm{acc}_{tr}(Q_b\theta)-\mathrm{acc}_{te}(Q_b\theta)]
-100[\mathrm{acc}_{tr}(\theta)-\mathrm{acc}_{te}(\theta)].
\]

Positive \(\Delta G\) means that quantization causes more *additional* damage to test accuracy than to training accuracy.

## Audited interpretation

Across the tested untargeted weight-only numerical perturbations, substantial low-precision accuracy loss is usually **joint**: training and test performance degrade together rather than producing a stable high-training/low-test post-grokking reversal. Numerical sensitivity is nevertheless heterogeneous across parameter groups; in the tested addition models, the SwiGLU gate is especially sensitive around four-bit integer quantization.

The repository also preserves historical executed notebooks and original notebook figures for provenance. Some notebook markdown, hypothesis labels, and source graphics predate the final audit and therefore contain interpretations that were later qualified or superseded. **For the current scientific interpretation, use the publication-facing figures, `docs/AUDIT_NOTES.md`, and the audited manuscript/supplement—not historical notebook prose.**

## Experimental scope

The archived experiment blocks include:

| Block | Training attempts / records | Reached recorded grokking criterion | Scope |
|---|---:|---:|---|
| E1 addition | 80 | 76 | \(p=97\): 40/40; \(p=59\): 36/40 |
| E5 multiplication | 50 | 47 | \(p=97\): 25/25; \(p=59\): 22/25 |
| E5 subtraction | 50 | 16 | \(p=97\): 14/25; \(p=59\): 2/25 |
| E6 width | 40 | 40 | Five feed-forward widths, eight model seeds each |
| E7 training fraction | 60 | 30 | Training fractions 0.2, 0.3, and 0.4 |

E1/E5 contain **23,730 repeated PTQ accuracy rows**. These rows are repeated measurements across precision, scheme, and checkpoint phase and must not be treated as 23,730 independent model replications.

## Repository layout

```text
.
├── notebooks/              Executed E1, E5, and consolidated analysis notebooks
├── data/
│   ├── raw_archives/       Original E1/E5/E6E7 experiment archives
│   ├── consolidated/       Curated consolidated experiment archive
│   └── processed/          Browseable consolidated CSV/JSON exports
├── results/tables/         Notebook-generated CSV and LaTeX tables
├── figures/
│   ├── main/               Four publication-facing main figures (PDF + PNG)
│   ├── supplementary/      Seven audited supplementary figures (PDF + PNG)
│   └── source_archive/     Historical/source notebook figure exports
├── docs/                   Experiment, data, provenance, and audit documentation
├── manifests/              File/figure/notebook/table inventories and SHA-256 checksums
├── scripts/                Archive extraction, validation, and checksum utilities
├── environment/            Reproduction environment specifications
└── .github/workflows/      Lightweight repository validation CI
```

## Notebooks

- `notebooks/e1_addition_experiments.ipynb` — executed E1 addition training and PTQ notebook.
- `notebooks/e5_multiplication_subtraction_experiments.ipynb` — executed E5 multiplication/subtraction replication notebook.
- `notebooks/grokked_transformer_quantization_analysis_v1.ipynb` — executed consolidated analysis notebook; it also contains the E6/E7 workflow and loads the E1/E5 exported datasets.
- `notebooks/exports/grokked_transformer_quantization_analysis_v1.py` — automatically exported Python representation of the consolidated notebook, retained for provenance and searchability. The `.ipynb` is the canonical executed record.

The notebooks were designed for Kaggle/Colab-style runtimes and preserve their original execution outputs. See `notebooks/README.md` before rerunning them locally.

## Data

Original archives:

```text
data/raw_archives/e1_dataset.zip
data/raw_archives/e5_dataset.zip
data/raw_archives/E6E7_DATASET_SLUG.zip
```

Curated consolidated archive:

```text
data/consolidated/grokked_transformer_quantization_experiments_v1.zip
```

For convenience, the consolidated CSV/JSON logs are also extracted under `data/processed/consolidated/`. See `data/README.md` and `docs/DATA_DICTIONARY.md`.

## Quantization operators

The archived evaluations contain three simulated weight-only numerical operators:

1. **Symmetric per-tensor integer quantization (INT).**
2. **Gaussian-quantile quantization (GQ).** The stored CSV label is `nf`; this is the notebook's normal-quantile implementation and **must not be interpreted as standard NF4**.
3. **Simulated mantissa rounding.** Equal nominal `bits` labels across operators do not imply equal physical storage budgets.

The `bits=23` condition is an explicit full-precision bypass for all three schemes.

## Main figures

<p align="center">
  <img src="figures/main/png/fig2_primary_selectivity.png" width="760" alt="Primary train-test selectivity figure">
</p>

The publication-facing main figures are available as vector PDF and PNG:

1. `fig1_grokking_trajectory`
2. `fig2_primary_selectivity`
3. `fig3_architectural_fragility`
4. `fig4_cross_task_robustness`

## Supplementary figures

The audited supplementary visual record contains:

1. `figS1_phenomenon`
2. `figS2_post_grok_instability`
3. `figS3_width_swiglu_verification`
4. `figS4_generality`
5. `figS5_width_theory`
6. `figS6_architecture_schematic`
7. `figS7_theory_verification`

Historical figures that are not part of the current publication evidence remain under `figures/source_archive/` for provenance. See `docs/FIGURE_PROVENANCE.md`.

## Reproducing the artifact

### 1. Create the environment

With Conda:

```bash
conda env create -f environment/environment.yml
conda activate grokked-quantization
```

or with pip:

```bash
python -m pip install -r environment/requirements.txt
```

The original long-running training experiments used a Tesla T4 with Python 3.12.13, PyTorch 2.10.0+cu128, NumPy 2.0.2, pandas 2.3.3, and SciPy 1.16.3. CPU execution is possible for analysis but is not a practical substitute for the original training runtime.

### 2. Validate the repository

```bash
python scripts/validate_repository.py
python scripts/verify_checksums.py
```

### 3. Extract archived data if needed

```bash
python scripts/extract_archives.py
```

The executed notebooks already contain outputs. Rerunning full E1/E5 training is optional and computationally expensive; secondary analysis can be performed from the archived numerical exports.

Detailed reproduction notes are in `docs/REPRODUCIBILITY.md`.

## Integrity and provenance

- `manifests/checksums_sha256.txt` provides SHA-256 hashes for repository artifacts.
- `manifests/source_inventory.csv` maps supplied source artifacts to their repository locations.
- `manifests/figure_manifest.csv` records publication status and provenance for figures.
- `manifests/notebook_manifest.csv` records the executed notebook roles.

No credentials or API tokens are required by the archived analyses.

## Citation

If you use this repository, please cite the accompanying manuscript and repository artifact. A machine-readable citation record is provided in `CITATION.cff`.

```bibtex
@misc{meherab2026grokquant,
  title  = {Train--Test Selectivity and Architectural Fragility Under Quantization of Grokked Transformers},
  author = {Meherab, Md Muntaqim and Muhammad, Mahathir},
  year   = {2026},
  note   = {Manuscript and accompanying reproducibility materials}
}
```

## Licenses

- Code and scripts: MIT License (`LICENSE`).
- Original research data and figures: CC BY 4.0 (`LICENSE-DATA.md`).

Before public release, verify that these terms match all institutional, co-author, and journal requirements.

## Contact

**Md Muntaqim Meherab**  
Daffodil International University  
Dhaka, Bangladesh  
Email: meherab2305101354@diu.edu.bd

For reproducibility questions, opening a GitHub issue is preferred so that answers remain visible to other users.
