# Phase 7: elastic pocket lifecycle
A local bounded state machine over the Phase 6 address space. Each pocket has a stable address and ID, active/retired state, capability allowlist, creation/change revision and generation. Retirement preserves the identity; restoration increments generation; a tombstone cannot be assigned to another pocket. At most 81 addresses and a separately configured active-pocket budget. Snapshots bind the nine-artifact manifest and policy; stale writers and invalid transitions fail closed. Snapshot validation is structural, not cryptographic authentication. The file example uses same-directory rename and a revision check, not a multi-process lock or crash-durable database.

## Reproduce
```sh
python training/train.py --profile nine --steps 300 --batch 8 --out runs/nine
node scripts/elastic-benchmark.mjs runs/nine
npm test
```
The benchmark refuses to overwrite an existing lifecycle snapshot. It uses a scripted three-phase workload, creates one scoped pocket per phase with the three required capabilities, executes actual local TNY2 inference, retires that pocket, then verifies restart and restoration. The fixed baseline keeps three full-capability pockets active and receives identical scoped queries and weights. Both must agree on answers/status. This demonstrates lifecycle and capability admission, not learned adaptive optimization; the workload is deliberately sequential and favors the scripted schedule. Peak active **logical** pockets, cache stats and output parity are reported separately. No physical RAM, throughput, energy or accuracy benefit is inferred.

## Delegation and boundaries
Capability discovery filters active state, domain and prefix scope. A local root may delegate to a single eligible pocket using a deterministic preferred-address policy; a zero fuel budget refuses execution. The receipt binds revision, pocket generation and artifact digest. No autonomous multi-hop agent, remote execution, cryptographic signature or cross-repository protocol is claimed. Actual adapter training, dynamic discovery from other processes, concurrent state storage, and real upstream NBG/ABL interoperability are future gates. Independent full-model replicas from Phase 5 are not LoRA adapters. Neural answers remain unverified and never write lifecycle state. CI uses smoke-trained weights, not quality-validated models.
