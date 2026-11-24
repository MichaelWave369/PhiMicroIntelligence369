"""Matched sampled-example controls; two capacities, augmentation, canonical prompts."""
import argparse,json,time
from pathlib import Path
import torch
from train import train
from generalization_data import make_sets
CONFIGS={"raw32":(32,2,4,64),"aug32":(32,2,4,64),"aug64":(64,2,4,64),"canonical32":(32,2,4,64)}
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--steps",type=int,default=600);p.add_argument("--batch",type=int,default=8);p.add_argument("--seed",type=int,default=369);p.add_argument("--device",choices=("cpu","cuda"),default="cpu");a=p.parse_args()
    if min(a.steps,a.batch)<1:p.error("positive training budget required")
    if a.out.exists() and any(a.out.iterdir()):p.error("fresh output directory required")
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    sets=make_sets();models={}
    for key,config in CONFIGS.items():
        dataset="raw" if key=="raw32" else "canonical" if key=="canonical32" else "augmented"
        models[key]=train(sets[dataset],config,a.steps,a.batch,a.seed,a.device,a.out/(key+".bin"))
        models[key]["dataset"]=dataset
    for k,rows in sets.items():(a.out/(k+".json")).write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    manifest=dict(schema=1,kind="generalization",seed=a.seed,steps=a.steps,batch=a.batch,device=a.device,torch=torch.__version__,models=models)
    (a.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
