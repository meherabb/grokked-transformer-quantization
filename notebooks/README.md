# Notebooks

This directory preserves the executed notebooks used to generate the project artifacts.

## Canonical executed notebooks

| File | Role |
|---|---|
| `e1_addition_experiments.ipynb` | E1 modular-addition training, checkpointing, PTQ sweeps, layer/group interventions, and exported E1 records |
| `e5_multiplication_subtraction_experiments.ipynb` | E5 multiplication and subtraction training/PTQ replication |
| `grokked_transformer_quantization_analysis_v1.ipynb` | Consolidated analysis notebook; loads E1/E5 exports and contains the E6/E7 and theory/capture workflow |

`exports/grokked_transformer_quantization_analysis_v1.py` is an automatically generated notebook export. It is retained for provenance and text search; the executed `.ipynb` is the canonical record.

## Important audit note

The notebooks are preserved as executed research records. Their markdown and saved outputs include historical H1--H8 working hypotheses and some analyses that were later re-audited, qualified, or excluded from the publication-facing evidence. They are intentionally not silently rewritten because doing so would destroy provenance.

Use `docs/AUDIT_NOTES.md`, `docs/FIGURE_PROVENANCE.md`, and the publication-facing figures for the current interpretation.

## Runtime behavior

The notebooks detect Kaggle, Google Colab, or local execution. Long-running training was performed on a Tesla T4. The E1/E5 notebooks contain checkpoint/resume logic. Local execution can be substantially slower.

Before rerunning the consolidated notebook, adjust its E1/E5 dataset paths to your local extracted archives as described in `docs/REPRODUCIBILITY.md`.
