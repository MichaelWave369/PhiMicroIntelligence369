# Phase 10: hierarchical learned residuals
Fixed, hand-authored parent groups: communication (network, coding, language), physical (science, units, astronomy), structure (geography, geometry, music). Train three rank-3 parents on their three domains first, freeze parent and shared base, then train nine rank-3 children with the correct parent active at both block exits. Parent and child see the same pre-adapter state, so deployment uses Phase 9's additive semantics. Parent FP32 digest is recorded in the child's manifest; INT8 inference uses the paired quantized parent, not exact training numerics. Neither the base nor parent parameters update during child training.

| Control | Total residual params | Active rank per known query | Adaptation samples |
|---|---:|---:|---:|
| Nested 3+9 rank-3 | 4608 | 6 | 18 × steps × batch |
| Flat nine rank-4 | 4608 | 4 | same |
| Pooled rank-36 | 4608 | 36 | same |
| Flat nine rank-6 | 6912 | 6 | same |

Base 23,744 parameters common to all; 28,352 with matched total residual count. Every control samples the same number of examples per domain (two times steps × batch), not identical examples or matched FLOPs. Rank-6 flat controls active rank but has more total parameters. Parent-only, child-only and wrong-parent ablations are not retrained; all comparisons use same dataset. Parent training groups are fixed hypotheses, not learned topology.

```sh
python training/train_nested_gears.py --out runs/nested369 --seed 369
node scripts/nested-benchmark.mjs runs/nested369
# Repeat with 636 and 963 in fresh directories.
node scripts/nested-aggregate.mjs runs/nested-aggregate.json runs/nested369 runs/nested636 runs/nested963
```
Training-only hierarchical trigram router first picks parent then a child within that group; compare with global flat trigram and lexical controls. No calibrated confidence. Evaluation includes 99 previously established held-out questions, exact match, per-split coverage, NLL on known answers, quantization, paired ablations and seed-level results. Because this dataset has been used throughout the project, it is not a fresh external confirmatory test. No test labels enter fitting. Results are a small pilot, not generalization to open-domain tasks.

PR CI: two-step plumbing plus separate **three-seed 60-step adaptation / 120-step base / batch-8 quality pilot**, with per-seed reports and aggregate artifact. Defaults for local run are 300 steps each. Pilot is exploratory; do not cherry-pick seeds or use the held-out results to claim a validated architecture. All neural outputs remain unverified. No true lazy memory/energy measurement or upstream NBG/ABL interop claimed.
