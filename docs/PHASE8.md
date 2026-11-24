# Phase 8: learned residual gears
A shared 23,744-parameter TNY2 transformer trained on the nine-domain training set, then frozen. Nine independently trained rank-4 linear residual output heads add B(Ah) to tied-embedding logits from final normalized hidden state h. Each head has 516 trainable parameters (4 × (32 + 97)); all nine total 4,644. The pooled rank-36 control has exactly 4,644 and the same total sampled adaptation examples. Total shared-base + residual parameters = 28,388 for both systems. Per-query routed activation uses 516 residual parameters; pooled uses 4,644. No internal-layer LoRA, weight streaming, NBG geometric proof, or matched FLOPs claim.

```sh
python training/train_gears.py --out runs/gears --base-steps 300 --steps 300 --batch 8 --seed 369
node scripts/gear-benchmark.mjs runs/gears
npm test
```
Run in fresh output directories for multiple seeds. Base is quantized/dequantized for frozen PyTorch gear training; JS inference additionally quantizes its KV cache, so numerical parity with training is not asserted. Export independent FP32 and row-scaled INT8 A/B factors (GRF1/GRQ1) with bound base SHA-256, shape, and strict length checks. INT8 parsing dequantizes to Float32Array; serialized byte reduction is not a physical RAM or compute reduction. Benchmark verifies actual file digests and metadata, then activates one residual at inference without mutating shared base. Registry retains all decoded gears during the benchmark; true lazy adapter residency remains future work.

Frozen 99-question evaluation: reference, frozen base, lexical-routed FP32 and INT8, fixed network INT8, intentionally swapped INT8, pooled FP32 and INT8. No training uses held-out subjects or templates. Compare exact match, coverage, unknown false-answer rate, per-domain/split metrics, latency rows, quantized output agreement and serialized payload. Route cue is hand-authored, not learned. All generated outputs remain unverified; blank neural outputs are not abstention. Equal total residual parameters and sample count do not imply equal FLOPs or capacity per query. No quality or improvement claim until independently repeated full training; CI uses two steps for plumbing only.

The gear experiment is an original conceptual NBG-inspired implementation at the output projection, separate from recursive topology. No changes to NBGPocketMesh or Altermath. Later gates: held-out learned route, intermediate-layer residuals, sparse multi-gear composition, calibration and budget-aware evaluation.
