"""Train a frozen shared TNY2 base and independent low-rank output-logit residuals."""
import argparse,hashlib,json,random,struct,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from dataset import build,NINE_DOMAINS
from train import Model,CONFIGS,VOCAB,batch,export,norm
D=32
RANK=4
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def describe(path,parameters,**kw):return dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,parameters=parameters,**kw)
def hidden(model,ids):
    x=model.emb(ids)+model.pos(torch.arange(ids.shape[1],device=ids.device))
    for b in model.blocks:x=b(x)
    return norm(x,model.nf)
def masked_loss(logits,ids,mask):
    loss=F.cross_entropy(logits.reshape(-1,VOCAB),ids[:,1:].reshape(-1),reduction="none")
    w=mask[:,1:].reshape(-1);return (loss*w).sum()/w.sum()
def fit_base(rows,steps,bsize,seed,device):
    torch.manual_seed(seed);rng=random.Random(seed);m=Model(CONFIGS["specialist"]).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=.003)
    for _ in range(steps):
        ids,mask=batch(rng.choices(rows,k=bsize),device);loss=masked_loss(m(ids[:,:-1]),ids,mask)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    return m,float(loss.detach().cpu())
def quantize_base_inplace(m):
    # Match TNY2 export row quantization; training still uses full-precision KV attention.
    tables=[m.emb.weight,m.pos.weight]
    for b in m.blocks:tables.extend(x.weight for x in (b.q,b.k,b.v,b.o,b.f1,b.f2))
    with torch.no_grad():
        for t in tables:
            scale=t.abs().amax(1,keepdim=True).div(127).clamp_min(1e-8)
            t.copy_((t/scale).round().clamp(-127,127)*scale)
    m.requires_grad_(False);m.eval()
class Gear(nn.Module):
    def __init__(self,rank):
        super().__init__();self.A=nn.Linear(D,rank,bias=False);self.B=nn.Linear(rank,VOCAB,bias=False)
        nn.init.normal_(self.A.weight,std=.03);nn.init.zeros_(self.B.weight)
    def forward(self,h):return self.B(self.A(h))
def fit_gear(base,rows,rank,steps,bsize,seed,device):
    torch.manual_seed(seed);rng=random.Random(seed);g=Gear(rank).to(device);opt=torch.optim.AdamW(g.parameters(),lr=.01)
    for _ in range(steps):
        ids,mask=batch(rng.choices(rows,k=bsize),device)
        with torch.no_grad():
            h=hidden(base,ids[:,:-1]);logits=h@base.emb.weight.T
        loss=masked_loss(logits+g(h),ids,mask)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(g.parameters(),1.0);opt.step()
    return g,float(loss.detach().cpu())
def write_gear(g,path,base_hash,quantized):
    rank=g.A.out_features
    with path.open("wb") as f:
        f.write((b"GRQ1" if quantized else b"GRF1")+struct.pack("<3H",D,VOCAB,rank)+bytes.fromhex(base_hash))
        for t in (g.A.weight,g.B.weight):
            for row in t.detach().cpu().float():
                if quantized:
                    scale=max(float(row.abs().max())/127,1e-8)
                    f.write(struct.pack("<f",scale));f.write(bytes(int(x)&255 for x in (row/scale).round().clamp(-127,127).to(torch.int8).tolist()))
                else:f.write(struct.pack("<"+"f"*len(row),*row.tolist()))
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--base-steps",type=int,default=300);p.add_argument("--steps",type=int,default=300);p.add_argument("--batch",type=int,default=8);p.add_argument("--seed",type=int,default=369);p.add_argument("--device",choices=("cpu","cuda"),default="cpu");a=p.parse_args()
    if min(a.steps,a.base_steps,a.batch)<1:p.error("positive steps and batch required")
    if a.out.exists() and any(a.out.iterdir()):p.error("output must be new or empty")
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows,test=build("nine");start=time.perf_counter()
    base,loss=fit_base(rows,a.base_steps,a.batch,a.seed,a.device)
    path=a.out/"base.bin";export(base,path);bh=sha(path)
    base_entry=describe(path,sum(p.numel() for p in base.parameters()),steps=a.base_steps,examples=a.base_steps*a.batch,loss=loss)
    quantize_base_inplace(base)
    # Check frozen base parameters do not change during residual training.
    frozen=[p.detach().clone() for p in base.parameters()]
    entries={}
    for i,d in enumerate((*NINE_DOMAINS,"pooled")):
        pooled=d=="pooled";rank=RANK*len(NINE_DOMAINS) if pooled else RANK
        steps=a.steps*len(NINE_DOMAINS) if pooled else a.steps
        g,loss=fit_gear(base,rows if pooled else [r for r in rows if r["domain"]==d],rank,steps,a.batch,a.seed+1+i,a.device)
        info=dict(rank=rank,steps=steps,examples=steps*a.batch,loss=loss,seed=a.seed+1+i)
        for fmt in ("fp32","int8"):
            path=a.out/(d+"."+fmt+".gear");write_gear(g,path,bh,fmt=="int8")
            info[fmt]=describe(path,sum(p.numel() for p in g.parameters()))
        entries[d]=info
    assert all(torch.equal(p,old) for p,old in zip(base.parameters(),frozen))
    assert sum(entries[d]["fp32"]["parameters"] for d in NINE_DOMAINS)==entries["pooled"]["fp32"]["parameters"]
    for name,items in (("train",rows),("test",test)):(a.out/(name+".json")).write_text(json.dumps(items,indent=2)+"\n",encoding="utf-8")
    manifest=dict(schema=1,kind="output-residual",seed=a.seed,device=a.device,torch=torch.__version__,base=base_entry,gears=entries,domains=list(NINE_DOMAINS),seconds=time.perf_counter()-start)
    (a.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
