"""Freeze distinct surface-form augmentation and fresh evaluation; no new facts in fit."""
import json
from dataset import ROOT,build,NINE_DOMAINS
from topology_partition import split_rows
def make_sets():
    raw,old=build("nine");fit,cal=split_rows(raw)
    templates=json.loads((ROOT/"data/augmentation.json").read_text())
    fresh=json.loads((ROOT/"data/fresh-evaluation.json").read_text())
    unique={(r["domain"],r["subject"]):r for r in fit}
    extra=[{**r,"question":t.format(subject=r["subject"])} for r in unique.values() for t in templates[r["domain"]]]
    aug=fit+extra
    canonical=[{**r,"question":r["domain"]+": "+r["subject"]} for r in unique.values()]
    # A domain- and subject-disjoint unknown case has no fact answer. No fresh row used for fitting.
    known=set(unique)
    assert len(fit)==108 and len(aug)==324 and len(canonical)==54 and len(fresh)==54
    assert len({r["question"].lower() for r in aug+canonical+cal+old+fresh})==len(aug+canonical+cal+old+fresh)
    for r in fresh:
        k=(r["domain"],r["subject"])
        assert (k in known)==(r["split"]=="fresh_known")
        assert (r["answer"] is None)==(r["split"]=="fresh_unknown")
        if k in known:assert r["answer"]==unique[k]["answer"]
    assert all(r["question"].isascii() and len(r["question"])<=72 and len(r["question"])+len(r["answer"] or "")+2<=128 for r in aug+canonical+cal+old+fresh)
    return {"raw":fit,"augmented":aug,"canonical":canonical,"calibration":cal,"old":old,"fresh":fresh}
