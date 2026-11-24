// Local experimental NBG/ABL-inspired typed execution, not an upstream runtime port.
import {classify} from "./mesh.mjs";
const ID=/^[a-z][a-z0-9_-]{0,31}$/;
const OPS=new Set(["query","sum","equal","join"]);
const MAX=18;
const copy=x=>JSON.parse(JSON.stringify(x));
export function validate(plan){
 if(!plan||!Array.isArray(plan.nodes)||plan.nodes.length<1||plan.nodes.length>MAX||!ID.test(plan.output))throw Error("invalid plan");
 const map=new Map();
 for(const n of plan.nodes){
  if(!n||typeof n.id!=="string"||!ID.test(n.id)||map.has(n.id)||!OPS.has(n.op))throw Error("invalid or duplicate node");
  if(!Array.isArray(n.deps)||n.deps.length>9||new Set(n.deps).size!==n.deps.length||n.deps.some(d=>typeof d!=="string"||!ID.test(d)))throw Error("invalid dependencies");
  if(n.op==="query"){if(n.deps.length||typeof n.question!=="string"||!n.question.trim()||n.question.length>72||!/^[ -~]+$/.test(n.question)||!classify(n.question))throw Error("invalid query")}
  else if(n.deps.length<2||n.op==="equal"&&n.deps.length!==2)throw Error("invalid transform arity");
  map.set(n.id,n);
 }
 if(!map.has(plan.output))throw Error("unknown output");
 const visiting=new Set(),seen=new Set();
 function visit(id){if(visiting.has(id))throw Error("cycle");if(seen.has(id))return;const n=map.get(id);if(!n)throw Error("missing dependency");visiting.add(id);n.deps.forEach(visit);visiting.delete(id);seen.add(id)}
 visit(plan.output);if(seen.size!==map.size)throw Error("unreachable nodes");
 return map;
}
function result(answer,status,origins=[]){return {answer,status,origins:[...new Set(origins)].sort()}}
function transform(n,inputs){
 const origins=inputs.flatMap(x=>x.origins);
 if(inputs.some(x=>x.answer===null))return result(null,"abstain",origins);
 const a=inputs.map(x=>x.answer),status=inputs.some(x=>x.status==="unverified")?"unverified":"reference-derived";
 if(n.op==="join")return result(a.join(" | "),status,origins);
 if(n.op==="equal")return result(String(a[0]===a[1]),status,origins);
 if(a.some(x=>! /^-?\d{1,6}$/.test(x)))return result(null,"abstain",origins);
 return result(String(a.reduce((s,x)=>s+Number(x),0)),status,origins);
}
function leaf(n,resolve){
 const domain=classify(n.question);
 const r=resolve(n.question,domain);
 if(!r||typeof r.answer!=="string"||!r.answer.trim()||r.answer.length>512||r.status==="abstain"||r.status==="error")return result(null,"abstain",[domain]);
 return result(r.answer.trim(),r.status==="reference-match"?"reference-derived":"unverified",[domain]);
}
export function execute(plan,resolve,{fuel=MAX,backend="belt"}={}){
 if(!Number.isInteger(fuel)||fuel<0||fuel>MAX||!["belt","flat"].includes(backend)||typeof resolve!=="function")throw Error("invalid execution policy");
 const nodes=validate(plan),done=new Map(),events=[];
 function run(n){
  if(done.size>=fuel)throw Error("fuel exhausted");
  const value=n.op==="query"?leaf(n,resolve):transform(n,n.deps.map(d=>done.get(d)));
  done.set(n.id,value);
  const seq=events.length+1;
  events.push({seq,previous:seq===1?null:seq-1,node:n.id,op:n.op,domain:n.op==="query"?classify(n.question):null,
    messages:n.deps.map(id=>({from:id,to:n.id,value:copy(done.get(id))})),value:copy(value)});
 }
 if(backend==="flat"){
  function visit(id){if(done.has(id))return;const n=nodes.get(id);n.deps.forEach(visit);run(n)}
  visit(plan.output);
 }else{
  while(done.size<nodes.size){
   const ready=[...nodes.values()].filter(n=>!done.has(n.id)&&n.deps.every(d=>done.has(d))).sort((a,b)=>a.id<b.id?-1:a.id>b.id?1:0);
   if(!ready.length)throw Error("no ready frontier");
   run(ready[0]);
  }
 }
 return {schema:1,backend,fuelUsed:done.size,output:copy(done.get(plan.output)),events};
}
// Navigation through recorded history, not inversion of non-invertible neural ops.
export function rewind(receipt,cursor){
 if(!Number.isInteger(cursor)||cursor<0||cursor>receipt.events.length)throw Error("invalid cursor");
 const events=receipt.events.slice(0,cursor);return {cursor,values:Object.fromEntries(events.map(e=>[e.node,copy(e.value)]))};
}
