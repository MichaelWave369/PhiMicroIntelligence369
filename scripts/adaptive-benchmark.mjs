import {readFileSync,writeFileSync} from "node:fs";import {join} from "node:path";import {createHash} from "node:crypto";
import {parseTiny,generate,TinySession,encode} from "../src/tiny.mjs";import {parseLayerGear,withLayerGears} from "../src/layer-gears.mjs";import {DOMAINS,classify,grade,metrics,reference} from "../src/mesh.mjs";import {chooseJunction,activation} from "../src/adaptive-junction.mjs";
const root=process.argv[2]||"runs/adaptive",read=n=>JSON.parse(readFileSync(join(root,n),"utf8")),m=read("manifest.json"),fit=read("fit.json"),cal=read("calibration.json"),test=read("test.json");
const variants=["fixed","learned","random"];if(m.schema!==1||m.kind!=="adaptive-topology"||m.partitionCount!==280||fit.length!==108||cal.length!==54||test.length!==99||fit.some(q=>q.split!=="train")||cal.some(q=>q.split!=="calibration")||new Set([...fit,...cal,...test].map(q=>q.question)).size!==261)throw Error("bad split/manifest");
function checked(e,file){if(e.file!==file)throw Error("bad filename");const b=readFileSync(join(root,file));if(b.length!==e.bytes||createHash("sha256").update(b).digest("hex")!==e.sha256)throw Error("digest mismatch");return b}
const base=parseTiny(checked(m.base,"base.bin"));if(base.parameters!==m.base.parameters)throw Error("base mismatch");
const registry={};for(const [id,e] of Object.entries(m.gears)){const g=parseLayerGear(checked(e,id+".int8.layer"));if(g.R!==e.rank||g.parameters!==e.parameters||g.baseHash!==m.base.sha256||g.D!==base.D||g.L!==base.L||g.format!=="int8")throw Error("gear metadata");registry[id]=g}
const budgets={};
function checkBudget(name,ids){const parameters=ids.reduce((s,id)=>s+m.gears[id].parameters,0),examples=ids.reduce((s,id)=>s+m.gears[id].examples,0);if(parameters!==4608||examples!==18*m.steps*m.batch)throw Error("budget mismatch "+name);budgets[name]={parameters,examples}}
for(const v of variants){const groups=m.groups[v];if(groups.length!==3||groups.some(g=>g.length!==3)||new Set(groups.flat()).size!==9||DOMAINS.some(d=>!groups.flat().includes(d)))throw Error("partition");const ids=[0,1,2].map(i=>v+"-p"+i).concat(DOMAINS.map(d=>v+"-c-"+d));checkBudget(v,ids);for(const d of DOMAINS){const p=v+"-p"+groups.findIndex(g=>g.includes(d)),c=m.gears[v+"-c-"+d];if(c.parent!==p||c.parentSha256!==m.gears[p].sha256)throw Error("parent binding")}}
checkBudget("flat",DOMAINS.map(d=>"flat-"+d));checkBudget("pooled",["pooled"]);
function model(ids){return ids.length?withLayerGears(base,ids.map(id=>({id,weight:1,gear:registry[id]})),m.base.sha256,{maxRank:ids[0]==="pooled"?36:6}):base}
function nll(model,q){if(q.answer===null)return null;const s=new TinySession(model);let logits;for(const t of [...encode(q.question),1])logits=s.step(t);const ids=[...encode(q.answer),1];let total=0;for(let i=0;i<ids.length;i++){const max=Math.max(...logits);total+=max+Math.log(logits.reduce((a,b)=>a+Math.exp(b-max),0))-logits[ids[i]];if(i+1<ids.length)logits=s.step(ids[i])}return total/ids.length}
const policy={};for(const v of variants){policy[v]={};for(const d of DOMAINS){const trained="p"+m.groups[v].findIndex(g=>g.includes(d));const c=chooseJunction(cal,d,["none","p0","p1","p2"],(p,q)=>nll(model(activation(v,d,p)),q));policy[v][d]={trained,...c}}}
// Freeze selection before evaluation. Test labels only enter scoring below.
writeFileSync(join(root,"junction-policy.json"),JSON.stringify({schema:1,seed:m.seed,policy},null,2)+"\n");
const systems=["reference","base","flat","pooled",...variants.flatMap(v=>[v+"-trained",v+"-adaptive",v+"-child-only"])],rows=[];
for(const [stage,questions] of [["fit",fit],["test",test]])for(const q of questions){
 const domain=classify(q.question);
 for(const system of systems){let ids=[],r,nn=null;
  if(system==="reference")r=reference(fit,q.question);
  else if(system!=="base"&&!domain&&system!=="pooled")r={answer:null,status:"abstain"};
  else{
   if(system==="flat")ids=["flat-"+domain];
   if(system==="pooled")ids=["pooled"];
   if(variants.some(v=>system.startsWith(v+"-"))){const v=variants.find(v=>system.startsWith(v+"-")),p=policy[v][domain];ids=activation(v,domain,system.endsWith("adaptive")?p.selected:system.endsWith("child-only")?"none":p.trained)}
   const active=model(ids);r=generate(active,q.question);nn=nll(active,q);
  }
  rows.push({stage,system,domain:q.domain,split:q.split,question:q.question,expected:q.answer,answer:r.answer,status:r.status,correct:grade(q.answer,r),ids,activeRank:ids.reduce((s,id)=>s+registry[id].R,0),nll:nn});
 }
}
const summary={};for(const stage of ["fit","test"]){summary[stage]={};for(const s of systems){const rr=rows.filter(r=>r.stage===stage&&r.system===s),known=rr.filter(r=>r.nll!==null);summary[stage][s]={...metrics(rr),meanNll:known.length?known.reduce((t,x)=>t+x.nll,0)/known.length:null,meanActiveRank:rr.reduce((t,x)=>t+x.activeRank,0)/rr.length,splits:Object.fromEntries([...new Set(rr.map(r=>r.split))].map(k=>[k,metrics(rr.filter(x=>x.split===k))]))}}}
const report={schema:1,seed:m.seed,steps:m.steps,batch:m.batch,baseSteps:m.base.steps,groups:m.groups,groupScores:m.groupScores,affinity:m.affinity,budgets,policy,summary,rows};writeFileSync(join(root,"adaptive-benchmark.json"),JSON.stringify(report,null,2)+"\n");console.log(JSON.stringify({seed:m.seed,groups:m.groups,groupScores:m.groupScores,policy,summary},null,2));
