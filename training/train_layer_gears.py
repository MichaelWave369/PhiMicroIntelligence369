"""Independent internal block-exit residuals with matched pooled control."""
import argparse,json,random,struct,time
from pathlib import Path
import torch
from torch import nn
from dataset import build,NINE_DOMAINS
from train import batch,export,norm,VOCAB
from train_gears import fit_base,quantize_base_inplace,masked_loss,describe,sha
D,L,RANK=32,2,4
class Pair(nn.Module):
    def __init__(self,rank):
        super().__init__();self.A=nn.Linear(D,rank,bias=False);self.B=nn.Linear(rank,D,bias=False)
        nn.init.normal_(self.A.weight,std=.03);nn.init.zeros_(self.B.weight)
    def forward(self,x):return self.B(self.A(x))
class LayerGear(nn.Module):
    def __init__(self,rank):
        super().__init__();self.blocks=nn.ModuleList(Pair(rank) for _ in range(L))
    def forward(self,base,ids):
        x=base.emb(ids)+base.pos(torch.arange(ids.shape[1],device=ids.device))
        for b,g in zip(base.blocks,self.blocks):
            x=b(x);x=x+g(x)
        return norm(x,base.nf)@base.emb.weight.T
def fit(base,rows,rank,steps,bsize,seed,device):
    torch.manual_seed(seed);rng=random.Random(seed);g=LayerGear(rank).to(device);opt=torch.optim.AdamW(g.parameters(),lr=.01)
    for _ in range(steps):
        ids,mask=batch(rng.choices(rows,k=bsize),device)
        loss=masked_loss(g(base,ids[:,:-1]),ids,mask)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(g.parameters(),1.0);opt.step()
    return g,float(loss.detach().cpu())
def write(g,path,bh,quant):
    with path.open("wb") as f:
        rank=g.blocks[0].A.out_features
        f.write((b"LRQ1" if quant else b"LRF1")+struct.pack("<3H",D,L,rank)+bytes.fromhex(bh))
        for b in g.blocks:
            for t in (b.A.weight,b.B.weight):
                for row in t.detach().cpu().float():
                    if quant:
                        scale=max(float(row.abs().max())/127,1e-8);f.write(struct.pack("<f",scale));f.write(bytes(int(x)&255 for x in (row/scale).round().clamp(-127,127).to(torch.int8).tolist()))
                    else:f.write(struct.pack("<"+"f"*len(row),*row.tolist()))
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--base-steps",type=int,default=300);p.add_argument("--steps",type=int,default=300);p.add_argument("--batch",type=int,default=8);p.add_argument("--seed",type=int,default=369);p.add_argument("--device",choices=("cpu","cuda"),default="cpu");a=p.parse_args()
    if min(a.steps,a.base_steps,a.batch)<1:p.error("positive steps required")
    if a.out.exists() and any(a.out.iterdir()):p.error("output must be new or empty")
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows,test=build("nine");start=time.perf_counter();base,loss=fit_base(rows,a.base_steps,a.batch,a.seed,a.device);path=a.out/"base.bin";export(base,path);bh=sha(path)
    entry=describe(path,sum(p.numel() for p in base.parameters()),steps=a.base_steps,examples=a.base_steps*a.batch,loss=loss)
    quantize_base_inplace(base);frozen=[p.detach().clone() for p in base.parameters()];gears={}
    for i,d in enumerate((*NINE_DOMAINS,"pooled")):
        pooled=d=="pooled";rank=RANK*(9 if pooled else 1);steps=a.steps*(9 if pooled else 1)
        g,loss=fit(base,rows if pooled else [r for r in rows if r["domain"]==d],rank,steps,a.batch,a.seed+i+1,a.device)
        info=dict(rank=rank,steps=steps,examples=steps*a.batch,loss=loss)
        for fmt in ("fp32","int8"):
            path=a.out/(d+"."+fmt+".layer");write(g,path,bh,fmt=="int8");info[fmt]=describe(path,sum(p.numel() for p in g.parameters()))
        gears[d]=info
    assert all(torch.equal(p,x) for p,x in zip(base.parameters(),frozen))
    assert sum(gears[d]["fp32"]["parameters"] for d in NINE_DOMAINS)==gears["pooled"]["fp32"]["parameters"]
    for name,items in (("train",rows),("test",test)):(a.out/(name+".json")).write_text(json.dumps(items,indent=2)+"\n",encoding="utf-8")
    (a.out/"manifest.json").write_text(json.dumps(dict(schema=1,kind="layer-residual",seed=a.seed,device=a.device,torch=torch.__version__,base=entry,gears=gears,domains=list(NINE_DOMAINS),seconds=time.perf_counter()-start),indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
