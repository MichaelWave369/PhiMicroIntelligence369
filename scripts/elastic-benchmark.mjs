import {readFileSync,writeFileSync,renameSync,existsSync,mkdirSync} from "node:fs";
import {join} from "node:path";import {createHash} from "node:crypto";import {DOMAINS} from "../src/mesh.mjs";import {parseTiny} from "../src/tiny.mjs";import {ArtifactCache} from "../src/clusters.mjs";import {ElasticMesh,memoryStore} from "../src/elastic-mesh.mjs";
const root=process.argv[2]||"runs/nine",read=n=>JSON.parse(readFileSync(join(root,n),"utf8")),manifest=read("manifest.json"),test=read("test.json"),out=join(root,"elastic");mkdirSync(out,{recursive:true});
const file=join(out,"snapshot.json");if(existsSync(file))throw Error("snapshot exists; choose a fresh run directory or move the old elastic snapshot");
const store={read:()=>existsSync(file)?JSON.parse(readFileSync(file,"utf8")):null,write:(next,expected)=>{if((store.read()?.revision??0)!==expected)throw Error("stale revision");const tmp=file+".tmp";writeFileSync(tmp,JSON.stringify(next,null,2)+"\n");renameSync(tmp,file)}};
const elastic=new ElasticMesh(manifest,store,{maxActive:3});
const fixed=new ElasticMesh(manifest,memoryStore(),{maxActive:3});
fixed.create({address:"0000"});fixed.create({address:"0100"});fixed.create({address:"0200"});
function cache(m){return new ArtifactCache(m.mesh,a=>{const b=readFileSync(join(root,a.file));if(createHash("sha256").update(b).digest("hex")!==a.sha256)throw Error("hash mismatch");return parseTiny(b)},Math.max(...[...m.mesh.artifacts.values()].map(a=>a.bytes))*2)}
const ec=cache(elastic),fc=cache(fixed),rows=[];
const qs=DOMAINS.map(d=>test.find(q=>q.domain===d&&q.split==="held_template"));if(qs.some(q=>!q))throw Error("missing held template");
for(let batch=0;batch<3;batch++){
 const address=["0000","0100","0200"][batch],scope=address.slice(0,2),domains=DOMAINS.slice(batch*3,batch*3+3);
 elastic.create({address,domains});
 for(const q of qs.slice(batch*3,batch*3+3)){
  const policy={preferred:address,scope,fuel:1};
  const a=elastic.delegate(q.question,ec,policy),b=fixed.delegate(q.question,fc,policy);
  if(a.status!==b.status||a.answer!==b.answer)throw Error("output divergence");
  rows.push({domain:q.domain,question:q.question,elastic:a,fixed:b,elasticActive:elastic.snapshot().pockets.filter(p=>p.status==="active").length,fixedActive:3});
 }
 elastic.retire(address);
}
const restarted=new ElasticMesh(manifest,store,{maxActive:3});const before=restarted.snapshot();if(JSON.stringify(before)!==JSON.stringify(elastic.snapshot()))throw Error("restart divergence");
restarted.restore("0000");const record=restarted.snapshot().pockets.find(p=>p.address==="0000");if(record.generation!==2||record.id!=="pocket:0000")throw Error("identity drift");
const blocked=restarted.delegate(qs[0].question,ec,{scope:"00",blockedPrefixes:["00"]});
if(blocked.status!=="abstain")throw Error("isolation failed");
const report={schema:1,rows,elasticCache:ec.stats(),fixedCache:fc.stats(),persistentRevision:restarted.snapshot().revision,restored:record,blocked,elasticPeakActive:Math.max(...rows.map(r=>r.elasticActive)),fixedPeakActive:3,scopeWorkload:"scripted sequential three-domain phases; no learned scheduling"};
writeFileSync(join(root,"elastic-benchmark.json"),JSON.stringify(report,null,2)+"\n");
console.log(JSON.stringify({queries:rows.length,outputParity:true,elasticPeakActive:report.elasticPeakActive,fixedPeakActive:3,restored:record,blocked:blocked.reason},null,2));
