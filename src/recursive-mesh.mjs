// Recursive, bounded ternary pocket addresses; nine shared TNY2 artifacts.
// The topology is logical, not a network, a neural geometry or a RAM measurement.
import {DOMAINS,classify} from "./mesh.mjs";
import {generate} from "./tiny.mjs";
export const DEPTH=4, MAX_POCKETS=81, SCALES=[1,4,9,41];
const integer=(x,min,max)=>Number.isSafeInteger(x)&&x>=min&&x<=max;
const hash=/^[a-f0-9]{64}$/;
export const pocketAddress=i=>{if(!integer(i,0,MAX_POCKETS-1))throw Error("invalid pocket index");return i.toString(3).padStart(DEPTH,"0")};
const nodeId=(p,d)=>p+"."+d;
function treeNode(prefix){return {prefix,children:new Map(),pocket:null}}
export function buildMesh(manifest,count){
 if(!integer(count,1,MAX_POCKETS)||manifest?.profile!=="nine"||!Array.isArray(manifest.domains)||manifest.domains.length!==9||DOMAINS.some(d=>!manifest.domains.includes(d)))throw Error("invalid mesh manifest or scale");
 const artifacts=new Map();
 for(const d of DOMAINS){
  const a=manifest.models?.[d];if(!a||a.file!==d+".bin"||!hash.test(a.sha256)||!integer(a.bytes,1,4_000_000)||!integer(a.parameters,1,4_000_000))throw Error("invalid artifact "+d);
  artifacts.set(d,{key:d,file:a.file,sha256:a.sha256,bytes:a.bytes,parameters:a.parameters});
 }
 const root=treeNode(""),pockets=new Map(),nodes=[],byId=new Map(),byDomain=new Map(DOMAINS.map(d=>[d,[]])),prefixes=new Set([""]);
 for(let i=0;i<count;i++){
  const address=pocketAddress(i),slots=new Map();let branch=root;
  for(const digit of address){if(!branch.children.has(digit))branch.children.set(digit,treeNode(branch.prefix+digit));branch=branch.children.get(digit);prefixes.add(branch.prefix)}
  for(const domain of DOMAINS){const n=Object.freeze({id:nodeId(address,domain),address,domain,artifact:domain,bytes:artifacts.get(domain).bytes});slots.set(domain,n);nodes.push(n);byId.set(n.id,n);byDomain.get(domain).push(n)}
  branch.pocket=slots;pockets.set(address,slots);
 }
 return {count,depth:DEPTH,root,pockets,nodes,byId,byDomain,prefixes,artifacts};
}
export function expandMesh(mesh,manifest,count){
 if(!integer(count,mesh.count,MAX_POCKETS))throw Error("expansion cannot shrink or exceed bound");
 const next=buildMesh(manifest,count);
 for(const [k,a] of mesh.artifacts){const b=next.artifacts.get(k);if(a.sha256!==b.sha256||a.bytes!==b.bytes||a.parameters!==b.parameters)throw Error("artifact drift during expansion")}
 return next;
}
export function routePocket(mesh,question,{preferred="0000",disabled=[],blockedPrefixes=[],maxArtifactBytes=4_000_000,maxDepth=DEPTH,backend="recursive"}={}){
 if(!mesh.pockets.has(preferred)||!["recursive","indexed","flat"].includes(backend)||!integer(maxArtifactBytes,0,4_000_000)||!integer(maxDepth,0,DEPTH)||!Array.isArray(disabled)||disabled.length>mesh.nodes.length||disabled.some(id=>!mesh.byId.has(id))||!Array.isArray(blockedPrefixes)||blockedPrefixes.length>mesh.prefixes.size||blockedPrefixes.some(p=>typeof p!=="string"||!mesh.prefixes.has(p)))throw Error("invalid route policy");
 const domain=classify(question),blocked=new Set(disabled),prefixSet=new Set(blockedPrefixes),stats={slotChecks:0,treeVisits:0,indexLookups:0,prefixChecks:0},trace=[];
 function isBlocked(address){for(let i=0;i<=address.length;i++){stats.prefixChecks++;if(prefixSet.has(address.slice(0,i)))return true}return false}
 const fail=reason=>({status:"abstain",domain,reason,stats,trace});
 const selected=n=>({status:"selected",domain,node:n,stats,trace});
 if(!domain)return fail("unclassified");
 if(maxDepth<mesh.depth)return fail("depth-budget");
 if(mesh.artifacts.get(domain).bytes>maxArtifactBytes)return fail("artifact-budget");
 function eligible(n){stats.slotChecks++;return !blocked.has(n.id)&&!isBlocked(n.address)}
 if(backend==="flat"){
  const hits=[];for(const n of mesh.nodes){stats.slotChecks++;if(n.domain===domain&&!blocked.has(n.id)&&!isBlocked(n.address))hits.push(n)}
  const n=hits.find(n=>n.address===preferred)??hits[0];return n?selected(n):fail("no eligible pocket");
 }
 if(backend==="indexed"){
  stats.indexLookups++;const p=mesh.byId.get(nodeId(preferred,domain));if(eligible(p))return selected(p);
  stats.indexLookups++;for(const n of mesh.byDomain.get(domain))if(n.address!==preferred&&eligible(n))return selected(n);
  return fail("no eligible pocket");
 }
 function visit(n){stats.treeVisits++;trace.push(n.prefix||"root");stats.prefixChecks++;return !prefixSet.has(n.prefix)}
 // Try the preferred path first, then lexicographic depth-first fallback.
 let branch=mesh.root,available=visit(branch);
 for(const digit of preferred){if(!available)break;branch=branch.children.get(digit);available=visit(branch)}
 if(available){const n=branch.pocket.get(domain);stats.slotChecks++;if(!blocked.has(n.id))return selected(n)}
 function search(n){
  if(!visit(n))return null;
  if(n.pocket){if(n.prefix===preferred)return null;const slot=n.pocket.get(domain);stats.slotChecks++;return blocked.has(slot.id)?null:slot}
  for(const child of n.children.values()){const hit=search(child);if(hit)return hit}
  return null;
 }
 const hit=search(mesh.root);return hit?selected(hit):fail("no eligible pocket");
}
export function queryPocket(mesh,cache,question,policy={},infer=generate){
 const r=routePocket(mesh,question,{...policy,maxArtifactBytes:Math.min(cache.budget,policy.maxArtifactBytes??4_000_000)});
 if(r.status!=="selected")return {answer:null,status:"abstain",route:r};
 const model=cache.acquire(r.node.artifact);if(!model)return {answer:null,status:"abstain",route:r};
 const result=infer(model,question);return {answer:result.answer,status:"unverified",route:r};
}
