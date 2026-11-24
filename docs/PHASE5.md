# Phase 5: 36 logical pockets / four clusters
Four groups of nine domain-specialist slots. Each query first selects a cluster preference (explicit benchmark input), then selects the eligible local domain node, falling back deterministically to other clusters when unavailable or over budget. Flat control scans every node under identical policy. No learned routing, distributed networking or autonomous delegation is claimed.

## Reproduce
```sh
python training/train.py --profile nine --steps 300 --batch 8 --out runs/nine
node scripts/cluster-benchmark.mjs runs/nine shared-nine
```
Shared-nine mode uses 36 logical nodes referencing nine real TNY2 artifacts; it does not pretend to contain 36 independently trained models. To train independent replicas:
```sh
python training/train_clusters.py --out runs/clusters36 --steps 300 --batch 8
node scripts/cluster-benchmark.mjs runs/clusters36 independent-36
```
The second mode trains 36 independent initializations on the same nine-domain data, four seeds per domain. It has 854,784 trainable parameters across files (36 × 23,744); unique model quality, diversity and performance remain unmeasured until the full run. No parameter-matched monolith claim for 36.

An optional fourth CLI argument sets the serialized-artifact residency budget in bytes (default two model payloads). LRU caches by artifact identity; shared mode deduplicates slots. Each load verifies SHA-256 before parsing, and model metadata after parsing. Budget tracks **serialized payload bytes represented by cached model objects**, not measured process heap, RAM, VRAM or peak allocation; JS GC and transient buffers are excluded. Timing includes routing, reads, parsing and inference. Flat and indexed controls use identical policies and independent cold caches. Indexed lookup counts and flat scan counts describe implementation work, not an experimentally measured computational speedup.

Benchmark: frozen 99 questions, explicit preferred cluster round robin, deterministic disabled-slot injection every fifth question, six causal composition plans with per-leaf cluster preferences. Both schedulers must agree on chosen nodes and output. Record results, routing inspection counts, cache hits/evictions, logical peak bytes and unique artifact count/digests. All model outputs remain unverified. CI exercises shared-nine with actual smoke-trained nine models, plus synthetic 36-artifact topology tests. Full independent training is opt-in. No NBG mathematical superiority or production readiness is claimed.
