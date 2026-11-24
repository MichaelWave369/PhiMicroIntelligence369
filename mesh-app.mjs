import {DOMAINS} from "./src/mesh.mjs";import {EvidenceMemory,arbitrate} from "./src/evidence.mjs";
const $=id=>document.getElementById(id),worker=new Worker("./src/mesh-worker.mjs",{type:"module"}),slots=[...DOMAINS,"monolith"];let memory=null,busy=false,question="";
const status=s=>$("status").textContent=s;
try{memory=new EvidenceMemory(localStorage);status("Local evidence loaded ("+memory.rows.length+" records)")}catch(e){status("Memory unavailable: "+e.message);$("save").disabled=true;$("clear").disabled=true}
const show=x=>$("result").textContent=JSON.stringify(x,null,2);
for(const d of slots){const option=document.createElement("option");option.value=d;option.textContent=d;$("slot").append(option)}
function lock(v){busy=v;$("ask").disabled=v;$("load").disabled=v;$("file").disabled=v;$("slot").disabled=v}
$("load").onclick=async()=>{if(busy)return;const f=$("file").files[0],domain=$("slot").value;if(!f){status("Choose a TNY2 binary");return}lock(true);try{const buffer=await f.arrayBuffer();worker.postMessage({type:"load",domain,buffer},[buffer])}catch(e){lock(false);status(e.message)}};
$("ask").onclick=()=>{if(busy)return;question=$("question").value.trim();if(!question||question.length>72||!/^[ -~]+$/.test(question)){status("Enter 1–72 printable ASCII characters");return}lock(true);status("Routing…");worker.postMessage({type:"ask",question})};
$("save").onclick=()=>{try{memory.add($("question").value,$("evidence").value,$("source").value);status("User-asserted evidence saved; not independently verified ("+memory.rows.length+")");$("evidence").value="";$("source").value=""}catch(e){status(e.message)}};
$("clear").onclick=()=>{if(confirm("Delete all evidence saved in this browser?"))try{memory.clear();status("Local evidence cleared")}catch(e){status(e.message)}};
const loaded=new Map();
worker.onmessage=({data:d})=>{lock(false);if(d.type==="loaded"){loaded.set(d.domain,{bytes:d.bytes,parameters:d.parameters});$("models").textContent=JSON.stringify(Object.fromEntries(loaded),null,2);status(d.domain+" loaded")}else if(d.type==="answer"){show({question,route:d.domain,decision:arbitrate(question,d.candidates,memory?.lookup(question)??[])});status("Complete; model output remains unverified")}else status(d.message)};
worker.onerror=e=>{lock(false);status("Worker error: "+e.message)};
