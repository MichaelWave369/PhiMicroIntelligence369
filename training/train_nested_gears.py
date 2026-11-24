"""Train parent gears first; freeze them while learning child residuals."""
import argparse,json,random,time
from pathlib import Path
import torch
from dataset import build,NINE_DOMAINS,ROOT
from train import batch,export,norm
from train_gears import fit_base,quantize_base_inplace,masked_loss,describe,sha
from train_layer_gears import LayerGear,write
GROUPS=json.loads((ROOT/"data/nested-groups.json").read_text(encoding="utf-8"))
def fit(base,rows,domains,rank,steps,bsize,seed,device,parent=None):
    torch.manual_seed(seed);rng=random.Random(seed);g=LayerGear(rank).to(device);opt=torch.optim.AdamW(g.parameters(),lr=.01)
    by_domain={d:[r for r in rows if r["domain"]==d] for d in domains}
    # Stratify update count exactly by domain in every control.
    assert steps%len(domains)==0
    for i in range(steps):
        ids,mask=batch(rng.choices(by_domain[domains[i%len(domains)]],k=bsize),device)
        x=base.emb(ids[:,:-1])+base.pos(torch.arange(ids.shape[1]-1,device=device))
        for j,(b,child) in enumerate(zip(base.blocks,g.blocks)):
            x=b(x);res=child(x)
            if parent is not None:res=res+parent.blocks[j](x)
            x=x+res
        loss=masked_loss(norm(x,base.nf)@base.emb.weight.T,ids,mask)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(g.parameters(),1.0);opt.step()
    return g,float(loss.detach().cpu())
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--base-steps",type=int,default=300);p.add_argument("--steps",type=int,default=300);p.add_argument("--batch",type=int,default=8);p.add_argument("--seed",type=int,default=369);p.add_argument("--device",choices=("cpu","cuda"),default="cpu");a=p.parse_args()
    if min(a.steps,a.base_steps,a.batch)<1:p.error("positive steps required")
    if a.out.exists() and any(a.out.iterdir()):p.error("output must be new or empty")
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows,test=build("nine");assert set(sum(GROUPS.values(),[]))==set(NINE_DOMAINS)
    start=time.perf_counter();base,loss=fit_base(rows,a.base_steps,a.batch,a.seed,a.device);path=a.out/"base.bin";export(base,path);bh=sha(path)
    base_entry=describe(path,sum(p.numel() for p in base.parameters()),steps=a.base_steps,examples=a.base_steps*a.batch,loss=loss)
    quantize_base_inplace(base);frozen=[p.detach().clone() for p in base.parameters()]
    gears={};parents={};parent_frozen={}
    def train_one(key,ds,rank,steps,seed,parent_key=None):
        g,loss=fit(base,rows,ds,rank,steps,a.batch,seed,a.device,parents.get(parent_key))
        info=dict(rank=rank,steps=steps,examples=steps*a.batch,loss=loss,seed=seed,domains=list(ds))
        if parent_key is not None:info.update(parent=parent_key,parentSha256=gears[parent_key]["fp32"]["sha256"])
        for fmt in ("fp32","int8"):
            path=a.out/(key+"."+fmt+".layer");write(g,path,bh,fmt=="int8");info[fmt]=describe(path,sum(p.numel() for p in g.parameters()))
        gears[key]=info;return g
    for i,(p,ds) in enumerate(GROUPS.items()):
        key="parent-"+p;g=train_one(key,ds,3,3*a.steps,a.seed+1+i);g.requires_grad_(False);g.eval();parents[key]=g;parent_frozen[key]=[v.detach().clone() for v in g.parameters()]
    for i,d in enumerate(NINE_DOMAINS):
        pk="parent-"+next(p for p,ds in GROUPS.items() if d in ds)
        train_one("child-"+d,[d],3,a.steps,a.seed+10+i,pk)
        train_one("flat-"+d,[d],4,2*a.steps,a.seed+30+i)
        train_one("active-"+d,[d],6,2*a.steps,a.seed+50+i)
    train_one("pooled",NINE_DOMAINS,36,18*a.steps,a.seed+70)
    assert all(torch.equal(p,x) for p,x in zip(base.parameters(),frozen))
    assert all(torch.equal(p,x) for k,g in parents.items() for p,x in zip(g.parameters(),parent_frozen[k]))
    for name,items in (("train",rows),("test",test)):(a.out/(name+".json")).write_text(json.dumps(items,indent=2)+"\n",encoding="utf-8")
    manifest=dict(schema=1,kind="nested-layer-residual",seed=a.seed,device=a.device,torch=torch.__version__,base=base_entry,gears=gears,groups=GROUPS,seconds=time.perf_counter()-start)
    (a.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
