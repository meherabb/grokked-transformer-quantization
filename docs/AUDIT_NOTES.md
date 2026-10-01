# Audit notes and interpretation boundary

The repository preserves executed research artifacts, including historical notebook prose and figures. The final manuscript audit introduced several important interpretation constraints.

## 1. Primary selectivity quantity

The current paper uses the **baseline-adjusted added gap**

`DeltaG = (quantized train-test gap) - (same-checkpoint FP32 train-test gap)`

in percentage points. A raw train-test gap is not interchangeable with DeltaG, particularly at the first-grokked checkpoint where the FP32 baseline may already have a non-trivial gap.

## 2. Independence

Bit widths, schemes, phases, and group interventions are repeated evaluations on trained models. Statistical resampling must operate at the model level when model-level uncertainty is intended.

## 3. Strong-reversion screen

The retrospective operational screen is quantized train accuracy >= 0.90 and quantized test accuracy <= 0.50. It occurs frequently in valid pre-grok snapshots because those states already have low test accuracy, but no post-grok model in the audited recorded sweep meets the screen at the first-grokked or cleaned endpoint.

## 4. Historical `nf` label

Stored `scheme="nf"` data correspond to the implemented Gaussian/normal-quantile operator (GQ), not standard NF4.

## 5. Historical figures excluded from current evidence

The following source figures are retained only under `figures/source_archive/`:

- `fig3_mechanism` — contains an incomplete predicted-vs-observed threshold panel and stronger mechanism language than the audited evidence supports.
- `fig6_train_fraction` — contains historical embedded labels that are not compatible with the final eligibility-qualified interpretation (0/20 models grokked at train fraction 0.2).
- `fig7_statistical_rigor` — contains an embedded equivalence statement that is not supported by the final cluster-level sensitivity analysis.

The original `fig8_grokking_trajectory` is superseded by the final publication-facing trajectory figure.

## 6. Mechanism interpretation

The raw annihilation-versus-degradation association is large (about rho=0.87), but both quantities vary strongly with precision. After controlling for bit width and modulus, network-level associations are near zero in the audited analysis. The separately retrained capture models also lack hashes proving identity with E1 checkpoints. Therefore, the repository does not present weight annihilation as an established causal mediation mechanism.

## 7. E7 eligibility

Training-fraction runs reach the grokking criterion in 0/20 cases at 0.2, 10/20 at 0.3, and 20/20 at 0.4. Low quantized test accuracy in a model that never grokked is not evidence of quantization-induced reversion.

## 8. Theory/capture provenance

Capture records use reused modulus/seed identifiers but are separately retrained. Any join to E1 accuracy records is an exploratory cross-source association rather than a same-checkpoint causal test.

## 9. Executed notebooks are provenance records

Historical notebook markdown is preserved rather than rewritten after the fact. Consequently, old H1--H8 labels, provisional verdicts, and older figure captions may not match the final audited manuscript. This is deliberate provenance, not an endorsement of superseded claims.
