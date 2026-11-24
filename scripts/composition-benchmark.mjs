import {readFileSync,writeFileSync} from "node:fs";import {join} from "node:path";import {createHash} from "node:crypto";import {parseTiny,generate} from "../src/tiny.mjs";import {reference,specialist,grade,DOMAINS} from "../src/mesh.mjs";import {execute} from "../src/composition.mjs";
const root=process.argv[2]||"runs/nine",read=n=>JSON.parse(readFileSync(join(root,n),"utf8")),manifest=read("manifest.json"),train=read("train.json"),cases=JSON.parse(readFileSync(new URL("../data/composition.json",import.meta.url),"utf8"));
if(manifest.profile!=="nine"||manifest.domains.length!==9||DOMAINS.some(d=>!manifest.domains.includes(d)))throw Error("nine-profile required");
const models={};for(const d of [...DOMAINS,"monolith"]){const e=manifest.models[d],bytes=readFileSync(join(root,d+".bin"));if(createHash("sha256").update(bytes).digest("hex")!==e.sha256)throw Error("bad hash: "+d);const m=parseTiny(bytes);if(m.parameters!==e.parameters||m.bytes!==e.bytes)throw Error("bad manifest");models[d]=m}
const resolvers={reference:q=>reference(train,q),mesh:q=>specialist(models,q),monolith:q=>generate(models.monolith,q)};
const rows=[];
for(const c of cases)for(const [system,resolve] of Object.entries(resolvers)){
 const start=performance.now(),belt=execute(c,resolve),ms=performance.now()-start;const t=performance.now(),flat=execute(c,resolve,{backend:"flat"}),flatMs=performance.now()-t;
 const normalized=r=>({answer:r.output.answer,status:r.output.answer===null?"abstain":r.output.status});
 const same=JSON.stringify(belt.output)===JSON.stringify(flat.output);if(!same)throw Error("backend divergence "+c.id);
 rows.push({id:c.id,split:c.split,expected:c.expected,system,output:belt.output,correct:grade(c.expected,normalized(belt)),same,ms,flatMs,receipt:belt});
}
const summary=Object.fromEntries(Object.keys(resolvers).map(s=>{const r=rows.filter(x=>x.system===s);return [s,{correct:r.filter(x=>x.correct).length,total:r.length,backendAgreement:r.filter(x=>x.same).length}]}));
writeFileSync(join(root,"composition.json"),JSON.stringify({schema:1,summary,rows},null,2)+"\n");console.log(summary);
