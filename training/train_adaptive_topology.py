"""Phase 11: choose groupings using fit-only base gradient cosine, then train gears."""
import argparse,json,random,time
from pathlib import Path
import torch
from dataset import build,NINE_DOMAINS,ROOT
from train import batch,export
from train_gears import fit_base,quantize_base_inplace,masked_loss,describe,sha
from train_layer_gears import write
from train_nested_gears import fit
from topology_partition import partitions,score,choose,split_rows
def affinity(base,rows,device):
    base.requires_grad_(True)
    params=list(base.parameters());vectors={}
    for d in NINE_DOMAINS:
        ids,mask=batch([r for r in rows if r["domain"]==d],device)
        loss=masked_loss(base(ids[:,:-1]),ids,mask)
        g=torch.cat([x.detach().flatten() for x in torch.autograd.grad(loss,params)])
        vectors[d]=g/g.norm().clamp_min(1e-12)
    base.requires_grad_(False)
    return {a:{b:float(torch.dot(vectors[a],vectors[b]).cpu()) for b in NINE_DOMAINS} for a in NINE_DOMAINS}
def canonical(groups):
    return tuple(sorted((tuple(sorted(g,key=NINE_DOMAINS.index)) for g in groups),key=lambda g:NINE_DOMAINS.index(g[0])))
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--base-steps",type=int,default=300);p.add_argument("--steps",type=int,default=120);p.add_argument("--batch",type=int,default=8);p.add_argument("--seed",type=int,default=369);p.add_argument("--device",choices=("cpu","cuda"),default="cpu");a=p.parse_args()
    if min(a.steps,a.base_steps,a.batch)<1:p.error("positive budgets required")
    if a.out.exists() and any(a.out.iterdir()):p.error("fresh output directory required")
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    raw,test=build("nine");rows,cal=split_rows(raw)
    assert not {r["question"] for r in rows+cal}&{r["question"] for r in test}
    start=time.perf_counter();base,loss=fit_base(rows,a.base_steps,a.batch,a.seed,a.device)
    path=a.out/"base.bin";export(base,path);bh=sha(path)
    base_entry=describe(path,sum(p.numel() for p in base.parameters()),steps=a.base_steps,examples=a.base_steps*a.batch,loss=loss)
    quantize_base_inplace(base);frozen=[p.detach().clone() for p in base.parameters()]
    sim=affinity(base,rows,a.device)
    learned,count=choose(NINE_DOMAINS,sim)
    fixed=canonical(json.loads((ROOT/"data/nested-groups.json").read_text()).values())
    random_group=random.Random(a.seed+1100).choice([g for g in partitions(NINE_DOMAINS) if g not in (fixed,learned)])
    variants={"fixed":fixed,"learned":learned,"random":random_group}
    entries={}
    def save(key,g,rank,steps,ds,loss,parent=None):
        path=a.out/(key+".int8.layer");write(g,path,bh,True)
        entries[key]=describe(path,sum(p.numel() for p in g.parameters()),rank=rank,examples=steps*a.batch,domains=list(ds),loss=loss,**({"parent":parent,"parentSha256":entries[parent]["sha256"]} if parent else {}))
    for variant,groups in variants.items():
        parents={}
        for i,ds in enumerate(groups):
            key=f"{variant}-p{i}";g,loss=fit(base,rows,ds,3,3*a.steps,a.batch,a.seed+1+i,a.device)
            save(key,g,3,3*a.steps,ds,loss);g.requires_grad_(False);g.eval();parents[i]=g
        old={i:[p.detach().clone() for p in g.parameters()] for i,g in parents.items()}
        for i,d in enumerate(NINE_DOMAINS):
            j=next(k for k,ds in enumerate(groups) if d in ds)
            g,loss=fit(base,rows,[d],3,a.steps,a.batch,a.seed+10+i,a.device,parents[j])
            save(f"{variant}-c-{d}",g,3,a.steps,[d],loss,f"{variant}-p{j}")
        assert all(torch.equal(p,x) for i,g in parents.items() for p,x in zip(g.parameters(),old[i]))
    for i,d in enumerate(NINE_DOMAINS):
        g,loss=fit(base,rows,[d],4,2*a.steps,a.batch,a.seed+30+i,a.device)
        save("flat-"+d,g,4,2*a.steps,[d],loss)
    g,loss=fit(base,rows,NINE_DOMAINS,36,18*a.steps,a.batch,a.seed+70,a.device);save("pooled",g,36,18*a.steps,NINE_DOMAINS,loss)
    assert all(torch.equal(p,x) for p,x in zip(base.parameters(),frozen))
    for name,items in (("fit",rows),("calibration",cal),("test",test)):(a.out/(name+".json")).write_text(json.dumps(items,indent=2)+"\n")
    manifest=dict(schema=1,kind="adaptive-topology",seed=a.seed,base=base_entry,groups=variants,affinity=sim,partitionCount=count,groupScores={k:score(v,sim) for k,v in variants.items()},gears=entries,steps=a.steps,batch=a.batch,seconds=time.perf_counter()-start)
    (a.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
if __name__=="__main__":main()
