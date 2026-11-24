import {readFileSync,writeFileSync} from "node:fs";
import {join} from "node:path";
import {createHash} from "node:crypto";
import {parseTiny,generate} from "../src/tiny.mjs";
import {DOMAINS,classify,specialist,reference,grade,metrics} from "../src/mesh.mjs";
const root=process.argv[2]||"runs",read=n=>JSON.parse(readFileSync(join(root,n),"utf8"));
const train=read("train.json"),test=read("test.json"),manifest=read("manifest.json");
if(manifest.schema!==1)throw Error("unsupported manifest");
const models={};
for(const d of [...DOMAINS,"monolith"]){
 const entry=manifest.models[d],bytes=readFileSync(join(root,d+".bin"));
 if(entry.file!==d+".bin"||createHash("sha256").update(bytes).digest("hex")!==entry.sha256)throw Error("model integrity check failed: "+d);
 const m=parseTiny(bytes);if(m.parameters!==entry.parameters||m.bytes!==entry.bytes)throw Error("manifest mismatch");models[d]=m;
}
const rows=[],splits=[...new Set(test.map(r=>r.split))];
for(const q of test){
 for(const system of ["reference","mesh","monolith"]){
   const start=performance.now();
   const r=system==="reference"?reference(train,q.question):system==="mesh"?specialist(models,q.question):generate(models.monolith,q.question);
   const ms=performance.now()-start;
   rows.push({system,domain:q.domain,route:classify(q.question),split:q.split,question:q.question,expected:q.answer,answer:r.answer,status:r.status,ms,correct:grade(q.answer,r)});
 }
}
const summary={};
for(const system of ["reference","mesh","monolith"]){const a=rows.filter(r=>r.system===system),lat=a.map(r=>r.ms).sort((a,b)=>a-b);summary[system]={...metrics(a),medianMs:lat[Math.floor((lat.length-1)*.5)],p95Ms:lat[Math.ceil(lat.length*.95)-1],splits:Object.fromEntries(splits.map(s=>[s,metrics(a.filter(r=>r.split===s))])),domains:Object.fromEntries(DOMAINS.map(d=>[d,metrics(a.filter(r=>r.domain===d))]))}}
const report={schema:1,manifest,routerAccuracy:test.filter(r=>classify(r.question)===r.domain).length/test.length,summary,rows};
writeFileSync(join(root,"benchmark.json"),JSON.stringify(report,null,2)+"\n");console.log(JSON.stringify({routerAccuracy:report.routerAccuracy,summary},null,2));
