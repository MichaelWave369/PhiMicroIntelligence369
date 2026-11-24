import {readFileSync,writeFileSync} from "node:fs";
import {join} from "node:path";
import {createHash} from "node:crypto";
import {parseTiny} from "../src/tiny.mjs";
import {grade} from "../src/mesh.mjs";
import {makeTopology,ArtifactCache,queryCluster,CLUSTERS} from "../src/clusters.mjs";
import {sharedManifest} from "../src/cluster-manifest.mjs";
import {execute} from "../src/composition.mjs";
const root=process.argv[2]||"runs/nine",mode=process.argv[3]||"shared-nine";
const read=n=>JSON.parse(readFileSync(join(root,n),"utf8"));
if(!["shared-nine","independent-36"].includes(mode))throw Error("unknown mode");
const manifest=mode==="shared-nine"?sharedManifest(read("manifest.json")):read("cluster-manifest.json");
if(manifest.mode!==mode)throw Error("mode mismatch");
const t=makeTopology(manifest),test=read("test.json"),plans=JSON.parse(readFileSync(new URL("../data/composition.json",import.meta.url),"utf8"));
const budget=Number(process.argv[4]??Math.max(...[...t.artifacts.values()].map(a=>a.bytes))*2);
function loader(a){const bytes=readFileSync(join(root,a.file));if(createHash("sha256").update(bytes).digest("hex")!==a.sha256)throw Error("hash mismatch "+a.key);return parseTiny(bytes)}
const results={};
for(const backend of ["hierarchy","flat"]){
 const cache=new ArtifactCache(t,loader,budget),rows=[];
 for(let i=0;i<test.length;i++){
  const q=test[i],preferred=CLUSTERS[i%4],disabled=i%5===0?[preferred+"."+q.domain]:[];
  const start=performance.now(),r=queryCluster(t,cache,q.question,{backend,preferred,disabled});
  rows.push({question:q.question,expected:q.answer,split:q.split,domain:q.domain,answer:r.answer,status:r.status,correct:grade(q.answer,r),node:r.route.node?.id??null,inspections:r.route.inspections,lookups:r.route.lookups,ms:performance.now()-start});
 }
 const compositions=plans.map(p=>{
  let i=0;const routes=[];
  const receipt=execute(p,q=>{const r=queryCluster(t,cache,q,{backend,preferred:CLUSTERS[i++%4]});routes.push({q,node:r.route.node?.id??null});return r});
  return {id:p.id,expected:p.expected,output:receipt.output,routes,receipt};
 });
 results[backend]={rows,compositions,cache:cache.stats(),correct:rows.filter(r=>r.correct).length,inspections:rows.reduce((s,r)=>s+r.inspections,0),lookups:rows.reduce((s,r)=>s+r.lookups,0)};
}
const a=results.hierarchy,b=results.flat;
const compare=r=>r.rows.map(x=>[x.answer,x.status,x.node]);
if(JSON.stringify(compare(a))!==JSON.stringify(compare(b))||JSON.stringify(a.compositions.map(x=>[x.output,x.routes]))!==JSON.stringify(b.compositions.map(x=>[x.output,x.routes])))throw Error("route or output divergence");
const report={schema:1,mode,logicalNodes:t.nodes.length,uniqueArtifacts:t.artifacts.size,uniqueDigests:new Set([...t.artifacts.values()].map(a=>a.sha256)).size,artifactBytes:[...t.artifacts.values()].reduce((s,a)=>s+a.bytes,0),uniqueArtifactParameters:[...t.artifacts.values()].reduce((s,a)=>s+a.parameters,0),budget,results};
writeFileSync(join(root,"cluster-benchmark.json"),JSON.stringify(report,null,2)+"\n");console.log(JSON.stringify({mode,logicalNodes:report.logicalNodes,uniqueArtifacts:report.uniqueArtifacts,uniqueDigests:report.uniqueDigests,accuracy:[a.correct,test.length],hierarchyInspections:a.inspections,flatInspections:b.inspections,hierarchyCache:a.cache,flatCache:b.cache},null,2));
