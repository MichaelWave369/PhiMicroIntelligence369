import {parseTiny,generate} from "./tiny.mjs";import {DOMAINS,classify} from "./mesh.mjs";
const models={};
self.onmessage=({data:d})=>{try{
 if(d.type==="load"){if(![...DOMAINS,"monolith"].includes(d.domain))throw Error("Unknown slot");const m=parseTiny(d.buffer);models[d.domain]=m;self.postMessage({type:"loaded",domain:d.domain,bytes:m.bytes,parameters:m.parameters})}
 else if(d.type==="ask"){const domain=classify(d.question),ids=[domain,"monolith"].filter(x=>x&&models[x]);const candidates=[];for(const id of ids){const start=performance.now();try{const r=generate(models[id],d.question);candidates.push({origin:id,answer:r.answer,status:r.status,ms:performance.now()-start})}catch(e){candidates.push({origin:id,status:"error",answer:null,error:e.message})}}self.postMessage({type:"answer",domain,candidates})}
 }catch(e){self.postMessage({type:"error",message:e.message})}};
