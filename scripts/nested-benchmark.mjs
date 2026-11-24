import {readFileSync,writeFileSync} from "node:fs";import {join} from "node:path";import {createHash} from "node:crypto";
import {parseTiny,generate,TinySession,encode} from "../src/tiny.mjs";import {parseLayerGear} from "../src/layer-gears.mjs";import {DOMAINS,classify,grade,metrics,reference} from "../src/mesh.mjs";import {fitRouter,rankDomains} from "../src/learned-router.mjs";import {validateHierarchy,selectNested,fitNestedRouter,routeNested} from "../src/nested-gears.mjs";
const root=process.argv[2]||"runs/nested",read=n=>JSON.parse(readFileSync(join(root,n),"utf8")),m=read("manifest.json"),train=read("train.json"),test=read("test.json"),groups=JSON.parse(readFileSync(new URL("../data/nested-groups.json",import.meta.url),"utf8"));
const contract=validateHierarchy(m,groups);
function checked(e,file){if(e.file!==file)throw Error("bad filename");const b=readFileSync(join(root,file));if(b.length!==e.bytes||createHash("sha256").update(b).digest("hex")!==e.sha256)throw Error("digest mismatch");return b}
const base=parseTiny(checked(m.base,"base.bin"));if(base.parameters!==m.base.parameters)throw Error("base mismatch");
const registry={};for(const id of contract.keys){registry[id]={};for(const fmt of ["fp32","int8"]){const e=m.gears[id][fmt],g=parseLayerGear(checked(e,id+"."+fmt+".layer"));if(g.R!==m.gears[id].rank||g.parameters!==e.parameters||g.format!==fmt||g.baseHash!==m.base.sha256||g.D!==base.D||g.L!==base.L)throw Error("gear mismatch");registry[id][fmt]=g}}
const nestedRouter=fitNestedRouter(train,groups),flatRouter=fitRouter(train),rows=[],routes=[];
const systems=["reference","base","nested-int8","nested-fp32","parent-only","child-only","wrong-parent","flat-int8","flat-active-int8","pooled-int8","learned-nested","learned-flat"];
function nll(model,q){if(q.answer===null)return null;const s=new TinySession(model);let logits;for(const t of [...encode(q.question),1])logits=s.step(t);const ids=[...encode(q.answer),1];let total=0;for(let i=0;i<ids.length;i++){const max=Math.max(...logits);total+=max+Math.log(logits.reduce((s,x)=>s+Math.exp(x-max),0))-logits[ids[i]];if(i+1<ids.length)logits=s.step(ids[i])}return total/ids.length}
for(const q of test){
 const domain=classify(q.question),hier=routeNested(nestedRouter,q.question),flat=rankDomains(flatRouter,q.question)[0]?.domain??null;
 routes.push({domain:q.domain,split:q.split,question:q.question,parentCorrect:hier.parent===contract.parents[q.domain],nestedCorrect:hier.domain===q.domain,flatCorrect:flat===q.domain,nested:hier,flat});
 for(const system of systems){
  let chosen=domain,mode="nested",format="int8",maxRank=6;
  if(system==="nested-fp32")format="fp32";
  if(system==="parent-only")mode="parent";
  if(system==="child-only")mode="child";
  if(system==="wrong-parent")mode="wrong-parent";
  if(system==="flat-int8")mode="flat";
  if(system==="flat-active-int8")mode="active";
  if(system==="pooled-int8"){mode="pooled";maxRank=36;chosen=DOMAINS[0]}
  if(system==="learned-nested")chosen=hier.domain;
  if(system==="learned-flat"){chosen=flat;mode="flat"}
  let model=base,ids=[],r;const start=performance.now();
  if(system==="reference")r=reference(train,q.question);
  else if(system==="base")r=generate(base,q.question);
  else if(!chosen)r={status:"abstain",answer:null};
  else{({model,ids}=selectNested(base,registry,m,chosen,{mode,format,maxRank}));r=generate(model,q.question)}
  const ms=performance.now()-start;
  rows.push({system,question:q.question,domain:q.domain,split:q.split,expected:q.answer,answer:r.answer,status:r.status,correct:grade(q.answer,r),ids,activeRank:ids.reduce((s,id)=>s+registry[id][format].R,0),nll:system==="reference"||r.status==="abstain"?null:nll(model,q),ms});
 }
}
const summary=Object.fromEntries(systems.map(s=>{const r=rows.filter(x=>x.system===s),known=r.filter(x=>x.nll!==null);return [s,{...metrics(r),meanNll:known.length?known.reduce((t,x)=>t+x.nll,0)/known.length:null,meanActiveRank:r.reduce((t,x)=>t+x.activeRank,0)/r.length,splits:Object.fromEntries([...new Set(test.map(q=>q.split))].map(k=>[k,metrics(r.filter(x=>x.split===k))]))}]}));
const contrasts={};for(const b of ["parent-only","child-only","wrong-parent","flat-int8","flat-active-int8","pooled-int8"]){const a=rows.filter(r=>r.system==="nested-int8"),other=rows.filter(r=>r.system===b),known=a.map((r,i)=>[r,other[i]]).filter(([x,y])=>x.nll!==null&&y.nll!==null);contrasts[b]={nllDelta:known.reduce((s,[x,y])=>s+x.nll-y.nll,0)/known.length,regressions:a.filter((r,i)=>!r.correct&&other[i].correct).length,improvements:a.filter((r,i)=>r.correct&&!other[i].correct).length}}
const report={schema:1,seed:m.seed,baseParameters:base.parameters,budgets:contract.parameters,adaptationExamples:contract.examples,summary,contrasts,routes,rows};
writeFileSync(join(root,"nested-benchmark.json"),JSON.stringify(report,null,2)+"\n");console.log(JSON.stringify({seed:m.seed,budgets:report.budgets,summary,contrasts,routes:{parent:routes.filter(r=>r.parentCorrect).length,nested:routes.filter(r=>r.nestedCorrect).length,flat:routes.filter(r=>r.flatCorrect).length,total:routes.length}},null,2));
