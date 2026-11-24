# Phase 2: matched-budget three specialists
The three models each have 23,744 trainable parameters (D=32, L=2, H=4, M=64). The monolith has 71,232 (D=32, L=8, H=4, M=60), exactly equal to the sum. Matching trainable parameters does not match serialized bytes, activation memory, FLOPs, or latency. The monolith receives three times the update count, matching total sampled training examples, not compute. Measure rather than assume efficiency.

## Run locally
From repo root, with Python 3.11+, Node 20+ and PyTorch:
```sh
python -m pip install torch
python training/train.py --steps 300 --batch 8 --seed 369 --out runs
node scripts/benchmark.mjs runs
```
CPU is default; CUDA is optional (`--device cuda`, compatible PyTorch required; deterministic CUDA operations may need CUBLAS_WORKSPACE_CONFIG=:4096:8 set before launch).
The output contains three independently initialized/trained TNY2 binaries, a matched monolith, training/test manifests and JSON results. No external weights are needed. All generated files are ignored by Git. The 2-step CI training run tests plumbing only, not model quality. Run longer experiments across multiple seeds before drawing conclusions.

## Dataset
24 short authored facts with reference links. 18 seen facts, 54 training questions; 18 held-template questions; 12 held-subject questions; 3 synthetic unknowns. Holdouts are withheld from both neural and deterministic reference training. The reference uses explicit relation keywords and exact subject tokens; the neural router uses the same relation keywords. These are deliberately simple templated tasks and not a realistic open-domain benchmark. It is not a semantic router. A correct answer to a held-out subject could be chance or extrapolation. Unknowns count as correct only on explicit abstention. Empty neural generations are not abstention. Accuracy is case-sensitive exact match after trimming; no keyword loopholes.
 
## Boundaries
No pretrained claims, no calibration, no inference-time shared memory, no validated NBG/Altermath advantage. Report router accuracy separately; measure per-system accuracy, coverage, selective accuracy, false answers on unknowns, measured latency and model byte totals. Use multiple seeds and report training time and sample count. Runtime overhead and peak memory remain future measurements. 
