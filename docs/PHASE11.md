# Phase 11: learned topology and calibration-selected junctions

## Prior result
[Phase 10 completed three-seed pilot](https://github.com/MichaelWave369/PhiMicroIntelligence369/actions/runs/45329840178) ([aggregate job](https://github.com/MichaelWave369/PhiMicroIntelligence369/actions/runs/45329840178/job/138228465844), [artifact](https://github.com/MichaelWave369/PhiMicroIntelligence369/actions/runs/45329840178/artifacts/14890799364)): nested INT8 exact match 14/297 (4.71%), flat 15/297 (5.05%), base 7/297 (2.36%), reference 189/297 (63.64%; includes unknown abstentions). Mean NLL nested 2.32036, flat 2.29149, wrong parent 2.75516. Correct vs wrong parent NLL contrast was negative in all three seeds. Nested did not outperform flat. These results motivate a new test, not a validated advantage.

## Protocol
Use only 108 original training rows for weight fitting and gradient affinity; reserve the third authored template per known subject (54 rows) for calibration; 99 original held-out rows stay separate. All splits share some subjects by design; held-subject rows are untouched. The reused test set is exploratory, not fresh confirmation. Compute normalized gradients of answer-token loss with respect to the shared base weights per domain after base quantization. Cosine similarity is a local heuristic, not semantic distance or guaranteed transfer. Enumerate all 280 unordered partitions into three triplets, maximize sum of nine within-group cosine similarities; compare hand grouping and seeded random distinct grouping. Same base, rank-3 parents and children, rank-4 flat and rank-36 pooled controls, same 4,608 residual parameters and 18×steps×batch adaptation examples per variant. Domain sample counts match; exact sample sequences, FLOPs and extra grouping/selection cost do not.

After training, for each domain and each topology, compare the child's trained parent, both alternative parents, and no parent on calibration NLL. Freeze selected parent before scoring test. No test labels used in fitting or selection. Child alone uses rank 3; parent+child rank 6; flat rank 4; pooled rank 36. Alternative parent activation is experimental, not a claim of parent interchangeability. This is offline calibration per domain, not per-query self-modification or live topology learning. Lexical domain cues used consistently. Report fit accuracy (memorization diagnostic) separately from test accuracy, per-split scores, NLL, active rank, full policy and affinity, with all seeds including failures. Compare fixed/learned/random trained junctions, each adaptive and child-only, flat, pooled, base, reference. No superiority or RAM claims.

```sh
python training/train_adaptive_topology.py --out runs/adaptive369 --seed 369
node scripts/adaptive-benchmark.mjs runs/adaptive369
# Repeat with 636 and 963, then aggregate:
node scripts/adaptive-aggregate.mjs runs/adaptive-aggregate.json runs/adaptive369 runs/adaptive636 runs/adaptive963
```
PR quality workflow runs 300 base steps, 120 adaptation multiplier, batch 8 across three seeds. Longer budget and smaller fit set differ from Phase 10; direct before/after comparison is confounded. Original TNY2 runtime and existing workflows unchanged. No physical NBG proof, no actual upstream integration. Future independent test dataset and actual resource profiling required.
