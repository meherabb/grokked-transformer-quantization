# Reproducibility guide

## Recommended use

For most users, reproducing the **secondary analyses** from archived numerical outputs is preferable to retraining every model. E1/E5 training is long-running; the executed notebooks and exported datasets are included so the experimental provenance remains inspectable.

## Original environment

The E1/E5 environment records report:

- Python 3.12.13
- PyTorch 2.10.0+cu128
- NumPy 2.0.2
- pandas 2.3.3
- SciPy 1.16.3 (from executed notebook output)
- Tesla T4 GPU
- Kaggle runtime for the archived E1/E5 runs

The final cached-data analysis package also records a CPU PyTorch environment. These are distinct executions and their runtimes should not be added together as training compute.

## Prepare local data

Run:

```bash
python scripts/extract_archives.py
```

This creates `data/extracted/` from the archived ZIP files without modifying the original archives.

## E1 and E5

The E1 and E5 notebooks detect Kaggle, Colab, or local execution. They include checkpointing/resume logic and export their own ZIP archives. On a local machine, output/checkpoint paths are redirected to local directories by their runtime detection logic.

## Consolidated analysis notebook

The consolidated notebook was written to load E1/E5 exported logs and to run/organize E6/E7 and theory/capture analyses. Before rerunning it outside the original environment, update its E1/E5 path configuration to point to the extracted local `outputs/logs/` directories.

The notebook is preserved as an executed provenance record. Historical working hypotheses/plots inside it may be superseded by the audit notes; do not infer the final paper's claims from notebook markdown alone.

## Validation

```bash
python scripts/validate_repository.py
python scripts/verify_checksums.py
```

`validate_repository.py` checks required files, notebook JSON validity, duplicate-extension accidents, and GitHub's 100 MB per-file limit. `verify_checksums.py` checks packaged file hashes against `manifests/checksums_sha256.txt`.

## Determinism

The notebooks seed Python, NumPy, and PyTorch, disable cuDNN benchmarking, and request deterministic PyTorch algorithms with warnings. Hardware/library differences can still affect floating-point execution and exact training trajectories.
