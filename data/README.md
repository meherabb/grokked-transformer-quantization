# Data and archived experiment outputs

This directory separates original archives from the curated consolidated release.

## `raw_archives/`

- `e1_dataset.zip` — original E1 addition records, including PTQ rows, fine-grid rows, layerwise rows, mechanism rows, summaries, trajectory information, and environment/config metadata.
- `e5_dataset.zip` — original E5 multiplication/subtraction PTQ, mechanism, summary, trajectory, and environment/config records.
- `E6E7_DATASET_SLUG.zip` — E6 width, E7 training-fraction, theory/capture outputs, notebook figures, and notebook-generated tables. This older archive is retained because it includes information not present in the later consolidated archive, including tensor-size-related records.

## `consolidated/`

- `grokked_transformer_quantization_experiments_v1.zip` — curated consolidated experiment archive supplied with the final analysis package.

## `processed/consolidated/`

Browseable extraction of the consolidated archive's `outputs/logs/` CSV/JSON files. These copies are included for convenience; their authoritative packaged counterpart is the consolidated ZIP.

## Independence warning

Rows indexed by bit width, quantization scheme, checkpoint phase, or architectural intervention are repeated measurements on the same trained model. They are not independent model replications.

See `docs/DATA_DICTIONARY.md` for column definitions and nomenclature.
