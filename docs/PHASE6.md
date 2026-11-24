# Phase 6: recursive pocket mesh
A bounded ternary tree of four-digit addresses, 0..80 (81 pockets maximum, 729 logical slots). The preregistered scale ladder is 1, 4, 9 and 41 pockets (9, 36, 81 and 369 logical nodes). Each pocket contains the same nine declared specialist capabilities. Expansion returns a new topology preserving old addresses and rejecting artifact drift. This is not an infinite network or additional trained intelligence.

## Reproduce
```sh
python training/train.py --profile nine --steps 300 --batch 8 --out runs/nine
node scripts/recursive-benchmark.mjs runs/nine
npm test
```
Three routing controls use identical selection: prefer requested pocket, else lowest eligible address. Flat scans slots; ordinary domain index checks direct preferred and sorted candidate list; recursive traversal follows a preferred path then lexicographic subtree traversal with prefix pruning. Compare separately counted slot checks, index lookups, tree visits, prefix checks, construction time and query time. No aggregation of unlike operations into a speedup number. Policy validation and index construction cost are not included in the counters. Indexes/tree are built locally and shared weights stay the same; benchmark times are descriptive, not controlled hardware measurements. Ordinary indexed lookup is expected to be competitive; a hierarchy is not inherently faster.

The routing suite uses nine held-template question texts at each scale, seven fixed conditions each: healthy, preferred slot disabled, blocked two-digit branch, all domain slots disabled, root blocked, byte budget zero and depth budget below four. 252 scenarios; all three must agree. All budgets are serialized artifact bytes or logical depth, not physical memory, latency guarantees or total fuel. Traversal is finite and bounded by tree size. The separate actual inference smoke uses ten queries per backend at each scale with independent LRU caches, SHA-256 validation and equal model weights. Counts and outputs are recorded, but CI's two-step training is not a quality evaluation.

Expansion is a rebuilt in-memory index, not concurrent mutation or distributed discovery. All 369 logical slots refer to nine artifacts (213,696 parameters total across artifacts). No 369 independently trained models, GPU savings, inter-repo interoperability or NBG mathematical proof. Future: measured index rebuild/update and memory costs, independent topology holdouts, sparse manifests with distinct adapters, real communication and causal delegation.
