"""Train fresh character transformers and export TNY2 weights. No upstream training data."""
import argparse, hashlib, json, random, struct, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from dataset import build,DOMAINS
VOCAB,CTX=97,128
# Matched trainable parameters: 3 * 23,744 == 71,232.
CONFIGS={"specialist":(32,2,4,64),"monolith":(32,8,4,60)}
def encode(s): return [1 if c=="\n" else ord(c)-30 for c in s]
class Block(nn.Module):
    def __init__(self,d,h,m):
        super().__init__();self.h=h
        self.n1=nn.Parameter(torch.ones(d));self.n2=nn.Parameter(torch.ones(d))
        self.q=nn.Linear(d,d,bias=False);self.k=nn.Linear(d,d,bias=False);self.v=nn.Linear(d,d,bias=False);self.o=nn.Linear(d,d,bias=False)
        self.f1=nn.Linear(d,m,bias=False);self.f2=nn.Linear(m,d,bias=False)
    def forward(self,x):
        b,t,d=x.shape
        z=norm(x,self.n1)
        def heads(y):return y.reshape(b,t,self.h,d//self.h).transpose(1,2)
        a=F.scaled_dot_product_attention(heads(self.q(z)),heads(self.k(z)),heads(self.v(z)),is_causal=True)
        x=x+self.o(a.transpose(1,2).reshape(b,t,d))
        return x+self.f2(F.relu(self.f1(norm(x,self.n2))))
def norm(x,g):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*g
class Model(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;d,l,h,m=config
        self.emb=nn.Embedding(VOCAB,d);self.pos=nn.Embedding(CTX,d)
        self.blocks=nn.ModuleList(Block(d,h,m) for _ in range(l));self.nf=nn.Parameter(torch.ones(d))
        nn.init.normal_(self.emb.weight,std=.05);nn.init.normal_(self.pos.weight,std=.05)
    def forward(self,ids):
        x=self.emb(ids)+self.pos(torch.arange(ids.shape[1],device=ids.device))
        for b in self.blocks:x=b(x)
        return norm(x,self.nf)@self.emb.weight.T
def batch(rows,device):
    seq=[];masks=[]
    for r in rows:
        q=encode(r["question"]+"\n");a=encode(r["answer"]+"\n");seq.append(q+a);masks.append([0]*len(q)+[1]*len(a))
    n=max(map(len,seq))
    ids=torch.tensor([s+[0]*(n-len(s)) for s in seq],device=device,dtype=torch.long)
    mask=torch.tensor([m+[0]*(n-len(m)) for m in masks],device=device,dtype=torch.float32)
    return ids,mask
def export(model,path):
    with path.open("wb") as f:
        f.write(b"TNY2"+struct.pack("<4H",*model.config))
        def table(t):
            for row in t.detach().cpu().float():
                scale=max(float(row.abs().max())/127,1e-8)
                f.write(struct.pack("<f",scale))
                f.write(bytes(int(x)&255 for x in (row/scale).round().clamp(-127,127).to(torch.int8).tolist()))
        def gains(t):
            a=t.detach().cpu().float().tolist();f.write(struct.pack("<"+"f"*len(a),*a))
        table(model.emb.weight);table(model.pos.weight)
        for b in model.blocks:
            gains(b.n1)
            for layer in (b.q,b.k,b.v,b.o):table(layer.weight)
            gains(b.n2);table(b.f1.weight);table(b.f2.weight)
        gains(model.nf)
def train(rows,config,steps,batch_size,seed,device,path):
    torch.manual_seed(seed);rng=random.Random(seed)
    model=Model(config).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.003)
    start=time.perf_counter()
    for i in range(steps):
        # uniform sampling with replacement; same total sample count across systems
        ids,mask=batch(rng.choices(rows,k=batch_size),device)
        logits=model(ids[:,:-1]);loss=F.cross_entropy(logits.reshape(-1,VOCAB),ids[:,1:].reshape(-1),reduction="none")
        weights=mask[:,1:].reshape(-1);loss=(loss*weights).sum()/weights.sum()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    export(model,path)
    return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
        parameters=sum(p.numel() for p in model.parameters()),config=list(config),steps=steps,examples=steps*batch_size,
        final_batch_loss=float(loss.detach().cpu()),seconds=time.perf_counter()-start,seed=seed)
def main():
    p=argparse.ArgumentParser();p.add_argument("--steps",type=int,default=300);p.add_argument("--batch",type=int,default=8);p.add_argument("--seed",type=int,default=369);p.add_argument("--device",choices=("cpu","cuda"),default="cpu");p.add_argument("--out",type=Path,default=Path("runs"));a=p.parse_args()
    if a.steps<1 or a.batch<1:p.error("steps and batch must be positive")
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True,exist_ok=True);train_rows,test=build()
    for name,rows in (("train",train_rows),("test",test)):(a.out/(name+".json")).write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    results={}
    for i,d in enumerate(DOMAINS):
        results[d]=train([r for r in train_rows if r["domain"]==d],CONFIGS["specialist"],a.steps,a.batch,a.seed+i,a.device,a.out/(d+".bin"))
    results["monolith"]=train(train_rows,CONFIGS["monolith"],3*a.steps,a.batch,a.seed,a.device,a.out/"monolith.bin")
    assert sum(results[d]["parameters"] for d in DOMAINS)==results["monolith"]["parameters"]
    (a.out/"manifest.json").write_text(json.dumps(dict(schema=1,device=a.device,torch=torch.__version__,models=results),indent=2)+"\n",encoding="utf-8")
    print(json.dumps(results,indent=2))
if __name__=="__main__":main()
