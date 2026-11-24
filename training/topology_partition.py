"""Dependency-free balanced grouping, deterministic and exhaustive."""
from itertools import combinations
def partitions(items):
    items=tuple(items)
    if not items:
        yield ()
        return
    first,*rest=items
    for pair in combinations(rest,2):
        group=(first,*pair)
        remain=tuple(x for x in rest if x not in pair)
        for tail in partitions(remain):yield (group,*tail)
def score(groups,affinity):
    return sum(affinity[a][b] for group in groups for a,b in combinations(group,2))
def choose(domains,affinity):
    candidates=list(partitions(domains))
    if len(candidates)!=280:raise ValueError("expected nine domains")
    return max(candidates,key=lambda g:score(g,affinity)),len(candidates)
def split_rows(rows):
    # Last authored training template per subject is calibration, not model fitting.
    by={}
    for r in rows:
        if r["split"]!="train":raise ValueError("training rows only")
        by.setdefault((r["domain"],r["subject"]),[]).append(r)
    if len(by)!=54 or any(len(rs)!=3 for rs in by.values()):raise ValueError("unexpected dataset")
    fit=[r for rs in by.values() for r in rs[:2]]
    cal=[{**rs[2],"split":"calibration"} for rs in by.values()]
    assert len(fit)==108 and len(cal)==54
    assert not {r["question"] for r in fit}&{r["question"] for r in cal}
    return fit,cal
