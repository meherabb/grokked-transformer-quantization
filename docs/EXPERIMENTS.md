# Experiment inventory

## E1 — addition

- Task: modular addition
- Primes: 97 and 59
- Training attempts: 80
- Recorded grokked models: 76 (40/40 at 97; 36/40 at 59)
- Default train fraction: 0.4
- Model seeds: 40 per prime
- Nominal training budget: 12,000 steps; adaptive safety cap: 30,000
- Post-grok checkpoints: first recorded grokking, +500, +1500, +4000, terminal/cleaned
- PTQ settings: `bits={2,3,4,5,6,8,10,12,16,23}` and schemes `int`, `mantissa`, `nf`

## E5 — multiplication and subtraction

- Multiplication: 50 attempts, 47 recorded grokked
- Subtraction: 50 attempts, 16 recorded grokked
- Primes: 97 and 59
- 25 seeds per task/prime condition
- Same nominal PTQ precision labels and checkpoint structure as E1

## E6 — feed-forward width

- Task: addition
- Widths: `d_ff={96,171,341,682,1024}`
- Eight model seeds per width
- 40 width runs total
- MLP-only integer interventions include 2-, 4-, and 5-bit settings in the audited width analysis

## E7 — training fraction

- Task: addition at p=97
- Training fractions: 0.2, 0.3, 0.4
- 20 runs per training fraction
- Grokking eligibility: 0/20 at 0.2, 10/20 at 0.3, 20/20 at 0.4

## Theory/capture records

The stored theory table used in the audited mechanism analysis contains 23 separately retrained grokked addition capture models (12 at p=97, 11 at p=59), 10 tensors per model, and 10 precision labels, yielding 2,300 tensor-bit observations. These captures cannot be verified as byte-identical to E1 checkpoints because state hashes/checkpoints were not archived.

## Model architecture

Default model:

- one causal-attention block
- one SwiGLU feed-forward block
- `d_model=128`
- 4 attention heads (head width 32)
- default `d_ff=341`
- no bias
- no layer normalization
- full-batch AdamW
- learning rate `1e-3`
- weight decay `1.0`
- betas `(0.9, 0.98)`
- epsilon `1e-8`

The grokking threshold is the first *recorded* test accuracy >= 0.90. The same fixed test split is used to locate this threshold and report subsequent checkpoint test accuracy; it is not an untouched checkpoint-selection-independent holdout.
