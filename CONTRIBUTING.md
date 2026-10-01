# Contributing

This repository is primarily a reproducibility artifact for a research manuscript.

For reproducibility problems, please open an issue and include the exact artifact path, environment, command or notebook cell, and observed error/output.

Changes to numerical results or publication-facing figures should be accompanied by:

1. the source data and script/notebook used to produce the change;
2. an explanation of whether the change affects manuscript claims;
3. regenerated checksums (`python scripts/generate_checksums.py`);
4. successful repository validation (`python scripts/validate_repository.py`).

Please do not overwrite original archived experiment files; add a new versioned artifact instead.
