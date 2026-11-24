# Phase 3: six-domain memory mesh
Six specialist slots: network, science, language, coding, geography, SI units. Three-profile behavior remains default. The expanded authored dataset has 48 facts, 108 training questions, 36 held-template questions, 24 held-subject questions and six unknown entities. It remains a toy benchmark.
 
## Run
```sh
python training/train.py --profile six --steps 300 --batch 8 --seed 369 --out runs/six
node scripts/benchmark.mjs runs/six
python -m http.server 8000
```
Open http://localhost:8000/mesh.html and load the generated binaries into their named slots. The six 23,744-parameter specialists total 142,464 parameters; the matched monolith is D=64,L=4,H=4,M=121 with 142,464 parameters. Equal sample counts, not equal FLOPs. No trained quality claim until multi-seed experiments.
 
The console runs on local workers; the selected specialist and optional monolith exchange candidate records through arbitration (not autonomous agent messaging). Exact normalized questions key the bounded, source-labelled localStorage evidence ledger. Evidence is manually asserted, not independently verified. Conflicting sources or disagreeing models without evidence abstain. Neural agreement is always unverified; model answers never write evidence. Memory is local to the browser and origin, not an encrypted database.
 
CI runs two update steps per specialist to validate training/export/evaluation, not accuracy. The existing three-domain smoke remains. Future: held-out multi-domain questions, cross-agent communication protocols, calibration, NBG/Altermath hypothesis tests, full benchmark runs and actual integration.
