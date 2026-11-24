import {readFileSync,writeFileSync} from "node:fs";import {join} from "node:path";import {createHash} from "node:crypto";import {parseTiny,generate} from "../src/tiny.mjs";import {parseGear,withGear} from "../src/gears.mjs";import {DOMAINS,classify,grade,metrics,reference} from "../src/mesh.mjs";
const root=process.argv[2]||"runs/gears",read=n=>JSON.parse(readFileSync(join(root,n),"utf8")),m=read("manifest.json"),train=read("train.json"),test=read("test.json");
if(m.schema!==1||m.kind!=="output-residual"||JSON.stringify(m.domains)!==JSON.stringify(DOMAINS))throw Error("invalid manifest");
function checked(e,file){if(e.file!==file)throw Error("unexpected file");const b=readFileSync(join(root,file));if(createHash("sha256").update(b).digest("hex")!==e.sha256||b.length!==e.bytes)throw Error("integrity mismatch");return b}
const base=parseTiny(checked(m.base,"base.bin"));if(base.parameters!==m.base.parameters)throw Error("base mismatch");
// Verify every artifact on admission. Inference activates only one gear at a time.
// Benchmark registry intentionally retains all decoded gears; no physical RAM claim.
const gears={};for(const d of [...DOMAINS,"pooled"]){gears[d]={};for(const fmt of ["fp32","int8"]){const e=m.gears[d][fmt],g=parseGear(checked(e,d+"."+fmt+".gear"));if(g.parameters!==e.parameters||g.format!==fmt||g.R!==m.gears[d].rank)throw Error("gear metadata mismatch");gears[d][fmt]=withGear(base,g,m.base.sha256)}}
const systems=["reference","base","routed-fp32","routed-int8","fixed-int8","swapped-int8","pooled-int8","pooled-fp32"],rows=[];
for(const q of test){
 const domain=classify(q.question),idx=DOMAINS.indexOf(domain);
 for(const system of systems){
  const start=performance.now();let r,active=null;
  if(system==="reference")r=reference(train,q.question);
  else if(system==="base")r=generate(base,q.question);
  else if(system.startsWith("routed")&&idx<0)r={answer:null,status:"abstain"};
  else{
   const fmt=system.endsWith("fp32")?"fp32":"int8";
   active=system.startsWith("pooled")?"pooled":system.startsWith("fixed")?DOMAINS[0]:system.startsWith("swapped")?DOMAINS[(idx+1+DOMAINS.length)%DOMAINS.length]:domain;
   r=generate(gears[active][fmt],q.question);
  }
  rows.push({system,question:q.question,domain:q.domain,split:q.split,expected:q.answer,answer:r.answer,status:r.status,correct:grade(q.answer,r),active,ms:performance.now()-start});
 }
}
const summary=Object.fromEntries(systems.map(system=>{const r=rows.filter(x=>x.system===system);return [system,{...metrics(r),splits:Object.fromEntries([...new Set(test.map(q=>q.split))].map(s=>[s,metrics(r.filter(x=>x.split===s))])),domains:Object.fromEntries(DOMAINS.map(d=>[d,metrics(r.filter(x=>x.domain===d))]))}]}));
const totals={baseParameters:base.parameters,baseBytes:base.bytes,routedParameters:DOMAINS.reduce((s,d)=>s+m.gears[d].int8.parameters,0),pooledParameters:m.gears.pooled.int8.parameters,routedFP32Bytes:DOMAINS.reduce((s,d)=>s+m.gears[d].fp32.bytes,0),routedINT8Bytes:DOMAINS.reduce((s,d)=>s+m.gears[d].int8.bytes,0),pooledINT8Bytes:m.gears.pooled.int8.bytes,activeRoutedParameters:m.gears[DOMAINS[0]].int8.parameters};
if(totals.routedParameters!==totals.pooledParameters)throw Error("unmatched residual parameter budget");
const f=rows.filter(r=>r.system==="routed-fp32"),i=rows.filter(r=>r.system==="routed-int8");const quantAgreement=f.filter((r,j)=>r.answer===i[j].answer&&r.status===i[j].status).length;
writeFileSync(join(root,"gear-benchmark.json"),JSON.stringify({schema:1,totals,quantAgreement:{matches:quantAgreement,total:f.length},summary,rows},null,2)+"\n");console.log(JSON.stringify({totals,quantAgreement,summary},null,2));
