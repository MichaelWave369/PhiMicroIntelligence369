# Phase 9: compositional internal-layer gears
Each independently learned gear applies a rank-4 residual B(Ax) at the exit of **both** transformer blocks. Nine gears × 512 parameters = 4,608 trainable residual parameters. Shared frozen base = 23,744; total = 28,352. Rank-36 pooled adapter has 4,608 parameters and matched total sampled adaptation examples. The base is row-quantized/dequantized for training; JavaScript additionally quantizes KV cache. No numerical parity claim.

```sh
python training/train_layer_gears.py --out runs/layer369 --seed 369
node scripts/layer-benchmark.mjs runs/layer369
```
Repeat at seeds 636 and 963 with distinct directories, then run `node scripts/layer-aggregate.mjs runs/layer369 runs/layer636 runs/layer963`. CI runs two updates per specialist and base only; full multi-seed quality evaluation is opt-in. No result inferred from the smoke.

LRF1/LRQ1 store FP32 or INT8 per-row factors, bound to exact base digest. Simultaneous adapters all read the same pre-adapter activation per block, and weighted residuals are summed once. Stable activation order, distinct IDs, shape binding and rank budget checks. No in-place base mutation. INT8 is decoded to FP32; registry retains all factors. Budget is active rank, not measured RAM or FLOPs.

A character-trigram multinomial naive Bayes router is fitted from training questions and domain labels only. It ranks known features; scores are not calibrated confidence. Evaluation compares frozen base, lexical FP32/INT8, learned top-1/top-2, lexical+wrong-domain pair, swapped single and matched pooled control. Pairs use equal 0.5 weights: attenuation is a confound, not a pure measure of interference. Report greedy exact match, answer-token teacher-forced NLL, pair-minus-single NLL, regressions/improvements, split metrics and route correctness. Unknown cases do not have NLL. All generations are unverified; blank generation is not abstention. Six existing structured composition plans compare per-leaf specialist activation against a single equal-weight group of required domain gears reused on all leaves. Plan supplies the domain cues, not learned natural-language planning. This tests composition compatibility, not autonomous reasoning.

Future: same-weight correct-only attenuation control, learned mixtures trained on independent multi-domain tasks, nested parent/child residuals, lazy residency, parity and larger evaluation. No NBG geometric proof or superiority claim.
