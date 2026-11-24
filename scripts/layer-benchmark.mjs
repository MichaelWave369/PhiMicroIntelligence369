import {readFileSync,writeFileSync} from "node:fs";import {join} from "node:path";import {createHash} from "node:crypto";
import {parseTiny,generate,TinySession,encode} from "../src/tiny.mjs";import {parseLayerGear,withLayerGears} from "../src/layer-gears.mjs";import {DOMAINS,classify,grade,metrics} from "../src/mesh.mjs";import {fitRouter,rankDomains} from "../src/learned-router.mjs";import {execute} from "../src/composition.mjs";
const root=process.argv[2]||"runs/layers",read=n=>JSON.parse(readFileSync(join(root,n),"utf8")),m=read("manifest.json"),train=read("train.json"),test=read("test.json");
if(m.kind!=="layer-residual"||m.schema!==1||JSON.stringify(m.domains)!==JSON.stringify(DOMAINS))throw Error("bad manifest");
function checked(e,file){if(e.file!==file)throw Error("bad filename");const b=readFileSync(join(root,file));if(b.length!==e.bytes||createHash("sha256").update(b).digest("hex")!==e.sha256)throw Error("digest mismatch");return b}
const base=parseTiny(checked(m.base,"base.bin"));if(base.parameters!==m.base.parameters)throw Error("base mismatch");
const gears={};for(const d of [...DOMAINS,"pooled"]){gears[d]={};for(const fmt of ["fp32","int8"]){const e=m.gears[d][fmt],g=parseLayerGear(checked(e,d+"."+fmt+".layer"));if(g.parameters!==e.parameters||g.R!==m.gears[d].rank||g.format!==fmt)throw Error("bad gear metadata");gears[d][fmt]=g}}
const router=fitRouter(train),rows=[],systems=["base","lexical-int8","lexical-fp32","learned-top1","learned-top2","attenuated-single","matched-pair","full-pair","swapped","pooled"];
function modelFor(ids,fmt="int8",weight=1,maxRank=8){return ids.length?withLayerGears(base,ids.map(id=>({id,weight,gear:gears[id][fmt]})),m.base.sha256,{maxRank}):base}
function nll(model,q){
 if(q.answer===null)return null;
 const session=new TinySession(model);let logits;for(const t of [...encode(q.question),1])logits=session.step(t);
 const ids=[...encode(q.answer),1];let total=0;
 for(let i=0;i<ids.length;i++){const max=Math.max(...logits);total+=max+Math.log(logits.reduce((s,x)=>s+Math.exp(x-max),0))-logits[ids[i]];if(i+1<ids.length)logits=session.step(ids[i])}
 return total/ids.length;
}
for(const q of test){
 const lex=classify(q.question),rank=rankDomains(router,q.question).map(x=>x.domain),other=DOMAINS[(DOMAINS.indexOf(lex)+1)%9];
 for(const system of systems){
  let ids=[],fmt="int8",weight=1,maxRank=8;
  if(system.startsWith("lexical")){ids=lex?[lex]:[];if(system.endsWith("fp32"))fmt="fp32"}
  if(system==="learned-top1")ids=rank.slice(0,1);
  if(system==="learned-top2"){ids=rank.slice(0,2);weight=.5}
  if(system==="attenuated-single"){ids=lex?[lex]:[];weight=.5}
  if(system==="matched-pair"){ids=lex?[lex,other]:[];weight=.5}
  if(system==="full-pair"){ids=lex?[lex,other]:[];weight=1}
  if(system==="swapped")ids=lex?[other]:[];
  if(system==="pooled"){ids=["pooled"];maxRank=36}
  const start=performance.now(),model=modelFor(ids,fmt,weight,maxRank),r=generate(model,q.question),ms=performance.now()-start;
  rows.push({system,domain:q.domain,split:q.split,question:q.question,expected:q.answer,answer:r.answer,status:r.status,correct:grade(q.answer,r),active:ids,activeRank:ids.reduce((s,d)=>s+gears[d][fmt].R,0),nll:nll(model,q),ms,learnedTop1:rank[0]??null});
 }
}
const summary=Object.fromEntries(systems.map(s=>{const r=rows.filter(x=>x.system===s),known=r.filter(x=>x.nll!==null);return [s,{...metrics(r),meanNll:known.reduce((t,x)=>t+x.nll,0)/known.length,splits:Object.fromEntries([...new Set(test.map(q=>q.split))].map(k=>[k,metrics(r.filter(x=>x.split===k))]))}]}));
const single=rows.filter(r=>r.system==="lexical-int8"),pair=rows.filter(r=>r.system==="matched-pair"),learned=rows.filter(r=>r.system==="learned-top1"),attenuated=rows.filter(r=>r.system==="attenuated-single"),fullPair=rows.filter(r=>r.system==="full-pair");
const interference={knownCount:single.filter(r=>r.nll!==null).length,meanPairMinusSingleNll:pair.reduce((s,r,i)=>s+(r.nll===null?0:r.nll-single[i].nll),0)/single.filter(r=>r.nll!==null).length,meanPairMinusAttenuatedNll:pair.reduce((s,r,i)=>s+(r.nll===null?0:r.nll-attenuated[i].nll),0)/single.filter(r=>r.nll!==null).length,meanFullPairMinusSingleNll:fullPair.reduce((s,r,i)=>s+(r.nll===null?0:r.nll-single[i].nll),0)/single.filter(r=>r.nll!==null).length,regressions:pair.filter((r,i)=>single[i].correct&&!r.correct).length,improvements:pair.filter((r,i)=>!single[i].correct&&r.correct).length};
const plans=JSON.parse(readFileSync(new URL("../data/composition.json",import.meta.url),"utf8"));
const composition=plans.map(p=>{const domains=[...new Set(p.nodes.filter(n=>n.op==="query").map(n=>classify(n.question)))].sort();const combined=modelFor(domains,"int8",1/domains.length,12);const separate=execute(p,q=>generate(modelFor([classify(q)]),q));const shared=execute(p,q=>generate(combined,q));return {id:p.id,expected:p.expected,domains,separate:separate.output,shared:shared.output, separateCorrect:grade(p.expected,{...separate.output,status:separate.output.answer===null?"abstain":separate.output.status}),sharedCorrect:grade(p.expected,{...shared.output,status:shared.output.answer===null?"abstain":shared.output.status})}});
const total=DOMAINS.reduce((s,d)=>s+gears[d].int8.parameters,0);if(total!==gears.pooled.int8.parameters)throw Error("unmatched gear budget");
const report={schema:1,seed:m.seed,baseParameters:base.parameters,gearParameters:total,learnedRouteCorrect:learned.filter(r=>r.learnedTop1===r.domain).length,routeTotal:learned.length,summary,interference,composition,rows};
writeFileSync(join(root,"layer-benchmark.json"),JSON.stringify(report,null,2)+"\n");console.log(JSON.stringify({seed:m.seed,summary,interference,learnedRouteCorrect:report.learnedRouteCorrect,composition},null,2));
