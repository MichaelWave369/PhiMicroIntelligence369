// One isolated process per model, descriptive RSS and timings (not total system memory).
import {readFileSync,writeFileSync} from "node:fs";import {join} from "node:path";import {createHash} from "node:crypto";
import {parseTiny,generate} from "../src/tiny.mjs";import {grade,metrics,reference} from "../src/mesh.mjs";import {createNormalizer,normalizeQuestion} from "../src/canonical-question.mjs";
const [root,system]=process.argv.slice(2),read=n=>JSON.parse(readFileSync(join(root,n+".json"),"utf8")),manifest=read("manifest"),train=read("raw"),norm=createNormalizer(train);
if(!["reference","normalized-reference","raw32","aug32","aug64","canonical32"].includes(system))throw Error("unknown system");
const before=process.memoryUsage();let model=null,loadMs=0,bytes=0,parameters=0;
if(manifest.models[system]){const e=manifest.models[system],start=performance.now(),b=readFileSync(join(root,system+".bin"));if(e.file!==system+".bin"||b.length!==e.bytes||createHash("sha256").update(b).digest("hex")!==e.sha256)throw Error("model digest");model=parseTiny(b);if(model.parameters!==e.parameters)throw Error("model parameters");loadMs=performance.now()-start;bytes=e.bytes;parameters=e.parameters}
const after=process.memoryUsage(),answers=new Map(train.map(r=>[r.domain+":"+r.subject,r.answer]));
function run(q){if(system==="reference")return reference(train,q);
 if(system==="normalized-reference"||system==="canonical32"){const n=normalizeQuestion(norm,q);if(n.status!=="matched")return {status:"abstain",answer:null,normalization:n};if(system==="normalized-reference")return {status:"reference-match",answer:answers.get(n.question.replace(": ",":")),normalization:n};return {...generate(model,n.question),normalization:n}}
 return generate(model,q);
}
for(const q of train.slice(0,3))run(q.question);
const rows=[];for(const stage of ["raw","old","fresh"]){for(const q of read(stage)){const t=performance.now(),r=run(q.question),ms=performance.now()-t;rows.push({stage,split:q.split,domain:q.domain,question:q.question,expected:q.answer,answer:r.answer,status:r.status,correct:grade(q.answer,r),ms,normalization:r.normalization??null})}}
const summary={};for(const stage of ["raw","old","fresh"]){const rr=rows.filter(r=>r.stage===stage),lat=rr.map(r=>r.ms).sort((a,b)=>a-b);summary[stage]={...metrics(rr),medianMs:lat[Math.floor((lat.length-1)/2)],p95Ms:lat[Math.ceil(.95*lat.length)-1],splits:Object.fromEntries([...new Set(rr.map(r=>r.split))].map(s=>[s,metrics(rr.filter(r=>r.split===s))]))}}
const result={system,seed:manifest.seed,steps:manifest.steps,batch:manifest.batch,summary,resources:{node:process.version,platform:process.platform,arch:process.arch,modelBytes:bytes,parameters,loadMs,before,after,end:process.memoryUsage(),maxRSSKiB:process.resourceUsage().maxRSS},rows};
writeFileSync(join(root,system+".report.json"),JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify({system,summary,resources:result.resources},null,2));
