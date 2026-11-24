"""Optional independent 36-model training. Same nine-domain corpus, four seeds per domain."""
import argparse,json,os
from pathlib import Path
import torch
from dataset import build,NINE_DOMAINS
from train import train,CONFIGS
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--steps",type=int,default=300);p.add_argument("--batch",type=int,default=8);p.add_argument("--seed",type=int,default=369);p.add_argument("--device",choices=("cpu","cuda"),default="cpu");a=p.parse_args()
    if a.steps<1 or a.batch<1:p.error("steps and batch must be positive")
    if a.out.exists() and any(a.out.iterdir()):p.error("output must be new or empty")
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows,test=build("nine");artifacts={};slots={}
    for i in range(4):
        for j,d in enumerate(NINE_DOMAINS):
            c="c"+str(i);key=c+"-"+d
            artifacts[key]=train([r for r in rows if r["domain"]==d],CONFIGS["specialist"],a.steps,a.batch,a.seed+i*9+j,a.device,a.out/(key+".bin"))
            slots[c+"."+d]=key
    for name,items in (("train",rows),("test",test)):
        (a.out/(name+".json")).write_text(json.dumps(items,indent=2)+"\n",encoding="utf-8")
    manifest=dict(schema=1,mode="independent-36",device=a.device,torch=torch.__version__,artifacts=artifacts,slots=slots)
    (a.out/"cluster-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
