import {readFileSync,writeFileSync} from "node:fs";import {join} from "node:path";import {createHash} from "node:crypto";import {parseTiny} from "../src/tiny.mjs";import {DOMAINS,grade} from "../src/mesh.mjs";import {ArtifactCache} from "../src/clusters.mjs";import {buildMesh,expandMesh,routePocket,queryPocket,SCALES,pocketAddress} from "../src/recursive-mesh.mjs";import {scenarios} from "../src/recursive-scenarios.mjs";
const root=process.argv[2]||"runs/nine",read=n=>JSON.parse(readFileSync(join(root,n),"utf8")),manifest=read("manifest.json"),test=read("test.json");
const questions=DOMAINS.map(d=>test.find(q=>q.domain===d&&q.split==="held_template"));if(questions.some(q=>!q))throw Error("missing fixture");
const backends=["flat","indexed","recursive"],report=[];
let mesh=buildMesh(manifest,1);
function loader(a){const b=readFileSync(join(root,a.file));if(createHash("sha256").update(b).digest("hex")!==a.sha256)throw Error("hash mismatch");return parseTiny(b)}
const signature=r=>[r.status,r.node?.id??null,r.reason??null];
for(const scale of SCALES){
 const start=performance.now();mesh=expandMesh(mesh,manifest,scale);const buildMs=performance.now()-start;
 const cases=scenarios(mesh,questions),routing={},neural={};
 const budget=Math.max(...[...mesh.artifacts.values()].map(a=>a.bytes))*2;
 for(const backend of backends){
  const rows=[];for(const c of cases){const t=performance.now(),r=routePocket(mesh,c.question,{...c.policy,backend});rows.push({kind:c.kind,question:c.question,domain:c.domain,signature:signature(r),stats:r.stats,ms:performance.now()-t})}
  const totals=Object.fromEntries(["slotChecks","treeVisits","indexLookups","prefixChecks"].map(k=>[k,rows.reduce((s,r)=>s+r.stats[k],0)]));
  routing[backend]={rows,totals};
  const cache=new ArtifactCache(mesh,loader,budget),outputs=[];
  // Nine real model calls per backend and scale; include one repeated domain at a second address.
  for(const [i,q] of [...questions,questions[0]].entries()){const preferred=pocketAddress(i%scale),t=performance.now(),r=queryPocket(mesh,cache,q.question,{preferred,backend});outputs.push({q:q.question,answer:r.answer,status:r.status,node:r.route.node?.id??null,correct:grade(q.answer,r),ms:performance.now()-t})}
  neural[backend]={outputs,cache:cache.stats()};
 }
 const routeSig=b=>routing[b].rows.map(r=>r.signature),neuralSig=b=>neural[b].outputs.map(r=>[r.answer,r.status,r.node]);
 for(const b of ["indexed","recursive"]){if(JSON.stringify(routeSig(b))!==JSON.stringify(routeSig("flat"))||JSON.stringify(neuralSig(b))!==JSON.stringify(neuralSig("flat")))throw Error("backend divergence at "+scale)}
 report.push({pockets:scale,logicalNodes:mesh.nodes.length,uniqueArtifacts:mesh.artifacts.size,buildMs,routing,neural});
}
const result={schema:1,mode:"shared-nine",artifactBytes:[...mesh.artifacts.values()].reduce((s,a)=>s+a.bytes,0),artifactParameters:[...mesh.artifacts.values()].reduce((s,a)=>s+a.parameters,0),report};
writeFileSync(join(root,"recursive-benchmark.json"),JSON.stringify(result,null,2)+"\n");
console.log(JSON.stringify(report.map(r=>({pockets:r.pockets,nodes:r.logicalNodes,routing:Object.fromEntries(backends.map(b=>[b,r.routing[b].totals])),parity:true})),null,2));
