# Φ MicroIntelligence369

Research prototype for tiny transformer inference, reference-backed specialist routing, and small-model experiments. Inspired by [Tiny 2.1](https://github.com/WollyMonjac/Tiny-2.1). This is a real TNY2 reader and inference implementation, not a simulated response generator.

## Quick start
Requires Node.js 20+ for tests and any static HTTP server for the browser. No npm dependencies.
- `npm test` runs synthetic format/forward-pass tests and deterministic fixtures.
- `python -m http.server 8000`, then visit http://localhost:8000.
- Select a local upstream TINY.BIN or fetch the pinned upstream weights in the UI. The remote file is not bundled here. Inference executes locally in a Web Worker.
- Compare greedy neural text against exact reference matches and bounded integer arithmetic. Unknown reference questions explicitly abstain.

## Architecture
`src/tiny.mjs`: TNY2 parser, signed 8-bit row weights, quantized KV cache, causal multi-head attention, RMSNorm, residual feedforward, tied output embedding, character decoding.
`src/router.mjs`: exact alias matching with source links and bounded integer arithmetic; no learned router yet.
`src/worker.mjs`: isolates inference from the UI.
`index.html` + `app.mjs`: static experiment console.
`tests/`: unit and fixture tests.

## Evidence discipline
Model probability is not factual confidence. Reference matching is not retrieval generalization. The upstream binary has not yet been verified for numerical parity against DOS or PyTorch. Synthetic tests do not establish parity, accuracy, or superiority. No NBG/Altermath backend or trained specialist mesh is claimed in v0.1.

## Roadmap
1. Compare upstream DOS/PyTorch outputs against browser runtime, pin binary hash and publish reproducible evaluation.
2. Train independently licensed network, science and language specialists; compare against exact lookup and a similarly budgeted monolith.
3. Measure accuracy, abstention, latency, resident memory, model bytes and energy on identical tasks; hold out subjects and templates.
4. Add external memory, uncertainty-aware arbitration, then test explicitly defined NBG/Altermath hypotheses under matched budgets.
5. Integrate validated runtimes into PhiOS, RetroVerse and PhiFractalCity.

[Upstream and licensing](docs/UPSTREAM.md) · [Experiment protocol](docs/EXPERIMENTS.md)

Code MIT. External weights and datasets retain upstream notices and applicable licenses.

## Phase 2: three-specialist experiment
See [matched-budget specialist training and evaluation](docs/PHASE2.md). Run `python training/train.py` then `node scripts/benchmark.mjs runs` to train and compare three independently initialized specialists, a parameter-matched monolith and a deterministic reference baseline.

## Phase 3: six-domain memory mesh
See [Phase 3](docs/PHASE3.md) and the browser console at `mesh.html`. Train with `--profile six` for six specialists and a parameter-matched monolith.
