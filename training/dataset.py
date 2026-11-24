"""Build fixed, disjoint examples from the authored facts. No ML dependencies."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOMAINS=("network","science","language")
SIX_DOMAINS=DOMAINS+("coding","geography","units")
NINE_DOMAINS=SIX_DOMAINS+("geometry","music","astronomy")
def domains_for(profile):
    if profile=="three":return DOMAINS
    if profile=="six":return SIX_DOMAINS
    if profile=="nine":return NINE_DOMAINS
    raise ValueError("unknown profile")
def build(profile="three"):
    spec=json.loads((ROOT/"data/specialists.json").read_text(encoding="utf-8"))
    train=[]; test=[]
    def add(dst,domain,subject,template,answer,source,split):
        dst.append(dict(domain=domain,subject=subject,question=template.format(subject=subject),answer=answer,source=source,split=split))
    for domain in domains_for(profile):
        v=spec[domain]
        for i,(subject,answer,source) in enumerate(v["facts"]):
            if i<6:
                for template in v["templates"]: add(train,domain,subject,template,answer,source,"train")
                add(test,domain,subject,v["heldTemplate"],answer,source,"held_template")
            else:
                for template in [v["templates"][0],v["heldTemplate"]]: add(test,domain,subject,template,answer,source,"held_subject")
        add(test,domain,v["unknown"],v["templates"][0],None,None,"unknown")
    assert len({x["question"] for x in train+test})==len(train)+len(test)
    assert not {(x["domain"],x["subject"]) for x in train}&{(x["domain"],x["subject"]) for x in test if x["split"]=="held_subject"}
    assert all(len(x["question"])<=72 and (len(x["question"])+len(x["answer"] or "")+2)<=128 and x["question"].isascii() and (x["answer"] or "").isascii() for x in train+test)
    return train,test
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=ROOT/"runs");p.add_argument("--profile",choices=("three","six","nine"),default="three");a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    for name,rows in zip(("train","test"),build(a.profile)):
        (a.out/(name+".json")).write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
        print(name,len(rows))
