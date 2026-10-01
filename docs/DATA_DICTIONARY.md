# Data dictionary

This file documents the main fields in the exported CSV records.

## Common PTQ-row fields

| Field | Meaning |
|---|---|
| `p` | Prime modulus (59 or 97 in the archived experiments) |
| `seed` | Model initialization/run seed within a configuration |
| `task` | `addition`, `multiplication`, or `subtraction` |
| `phase` | Checkpoint phase such as `pre_grok`, `just_grokked`, `post_grok_500`, `post_grok_1500`, `post_grok_4000`, or `cleaned_up` |
| `phase_step` | Recorded training step for the checkpoint |
| `fp32_train_acc` | Full-precision training accuracy at the same checkpoint |
| `fp32_test_acc` | Full-precision test accuracy at the same checkpoint |
| `weight_norm` | Saved model-weight norm summary |
| `scheme` | Stored quantizer label: `int`, `mantissa`, or `nf` |
| `bits` | Nominal precision label. `23` is the explicit full-precision bypass |
| `q_train_acc` | Training accuracy after the numerical perturbation |
| `q_test_acc` | Test accuracy after the numerical perturbation |
| `d_ff` | Feed-forward width |
| `train_frac` | Fraction of the enumerated modular-arithmetic input space used for training |
| `tag` | Experiment-block tag |

### Important nomenclature: `nf`

The stored label `nf` refers to the project's **Gaussian-quantile (GQ)** / normal-quantile operator. It uses midpoint normal quantiles with a codebook size capped at 256 levels. It is **not** the normalized NF4 implementation used by QLoRA, despite the historical internal label.

## Fine-grid rows

Fields include `n_levels`, full-precision accuracies, quantized accuracies, `p`, `seed`, task, training fraction, and tag. For the archived level scan, labels 7--15 map to effective `q_max` values 3,3,4,4,5,5,6,6,7.

## Architectural-intervention rows

`group` identifies the isolated parameter group under quantization. Broad groups include embedding, attention, MLP, and unembedding. Projection-level records include SwiGLU gate/up/down interventions.

## Mechanism rows

Fields such as `erasure_cleanup` and `erasure_fulltrain` are historical weight-zeroing summaries under integer quantization. They should be interpreted with the audit qualifications in `docs/AUDIT_NOTES.md`.

## Theory/capture files

- `theory_ab_table.csv`: tensor-level integer zeroing fraction `A_b` by model, parameter tensor, group, and precision.
- `theory_activation_summary.csv`: gate/up activation magnitude summaries.
- `theory_width_maxmag_table.csv`: tensor peak/median magnitude and tensor element count across width runs.

## Run-summary files

Run summaries include grokking status/step, final accuracies, training fraction, width, actual steps used, convergence flags, and saved post-grokking dip counts.
