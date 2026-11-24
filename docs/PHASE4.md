# Phase 4: nine nodes and causal composition
Nine specialists × 23,744 = 213,696 trainable parameters. Matched monolith: D=96,L=4,H=4,M=57 (213,696). Identical sample count, not equal FLOPs or resident bytes. 72 authored facts, 162 training questions, 99 evaluation questions. New domains geometry, music intervals and planetary order. Three and six profiles remain supported.

## Run
```sh
python training/train.py --profile nine --steps 300 --batch 8 --seed 369 --out runs/nine
node scripts/benchmark.mjs runs/nine
node scripts/composition-benchmark.mjs runs/nine
```
The compositional suite is a tiny frozen, explicitly structured set of six tasks. It compares the exact reference baseline, actual specialist inference and actual monolith inference using the same plan, so the neural models do not have to generate plans. Four known/held-template tasks, one held subject, one unknown. The reference intentionally lacks the held subject. Unknown success requires abstention. No open-ended reasoning claim.

## NBG/Altermath adapter E0
Inspired by [NBGPocketMesh](https://github.com/MichaelWave369/NBGPocketMesh) bounded capability routing and [Altermath](https://github.com/MichaelWave369/Altermath) ABL causal belt. This is an original local implementation of a compatible *conceptual* pattern: typed query nodes, allowlisted transforms, causal dependencies, deterministic ready-frontier ordering, finite fuel, and directed candidate messages. No upstream runtime import, protocol compatibility, network authority, neural weight geometry, parallelism or mathematical proof. Rewind is inspection of recorded states, not inversion of computation. Compare a recursive flat scheduler with the causal belt on the same graph; identical outputs are expected and do not imply performance advantage. Trace receipts are ordered but not signed or hash-chained. Browser mesh includes nine model slots; compositional execution currently CLI/API only.

Future: pinned interop against actual upstream contracts, blinded natural-language planning with strict schema validation, larger independently designed composition cases, memory provenance and runtime budgets. Report all failures. CI uses two updates per specialist to check plumbing only.
