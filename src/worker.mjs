import {parseTiny,generate} from "./tiny.mjs";
let model=null;
self.onmessage=({data:d})=>{try{if(d.type==="load"){model=null;model=parseTiny(d.buffer);const {D,L,H,M,bytes,parameters}=model;self.postMessage({type:"loaded",meta:{D,L,H,M,bytes,parameters}})}else if(d.type==="generate"){if(!model)throw Error("Load a model first");const start=performance.now(),r=generate(model,d.question,55,token=>self.postMessage({type:"token",token}));self.postMessage({type:"done",answer:r.answer,ms:performance.now()-start})}}catch(e){self.postMessage({type:"error",message:e.message,loaded:!!model})}};
