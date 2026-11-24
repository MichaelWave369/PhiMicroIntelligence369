// Four bounded pockets, nine specialist capabilities per pocket.
// Resource accounting is serialized artifact bytes, not physical heap/VRAM.
import {DOMAINS,classify} from "./mesh.mjs";
import {generate} from "./tiny.mjs";
export const CLUSTERS=["c0","c1","c2","c3"];
const HASH=/^[a-f0-9]{64}$/;
const int=(x,min,max)=>Number.isSafeInteger(x)&&x>=min&&x<=max;
const cmp=(a,b)=>a<b?-1:a>b?1:0;
export function makeTopology(manifest){
 if(!manifest||!["shared-nine","independent-36"].includes(manifest.mode)||!manifest.artifacts||!manifest.slots)throw Error("invalid manifest");
 const nodes=[],artifacts=new Map(),clusters=new Map(CLUSTERS.map(c=>[c,new Map()]));
 for(const c of CLUSTERS)for(const d of DOMAINS){
  const id=c+"."+d,key=manifest.slots[id],a=manifest.artifacts[key];
  if(typeof key!=="string"||!/^[a-z0-9_-]+$/.test(key)||!a||a.file!==key+".bin"||!HASH.test(a.sha256)||!int(a.bytes,1,4_000_000)||!int(a.parameters,1,4_000_000))throw Error("bad artifact for "+id);
  if(manifest.mode==="shared-nine"&&key!==d||manifest.mode==="independent-36"&&key!==c+"-"+d)throw Error("invalid mapping "+id);
  const entry={key,file:a.file,sha256:a.sha256,bytes:a.bytes,parameters:a.parameters};artifacts.set(key,entry);
  const node={id,cluster:c,domain:d,artifact:key,bytes:a.bytes};nodes.push(node);clusters.get(c).set(d,node);
 }
 if(Object.keys(manifest.slots).length!==36||Object.keys(manifest.artifacts).length!==artifacts.size)throw Error("unexpected manifest entries");
 if(artifacts.size!==(manifest.mode==="shared-nine"?9:36))throw Error("wrong artifact count");
 return {mode:manifest.mode,nodes,clusters,artifacts};
}
export function routeCluster(topology,question,{preferred="c0",disabled=[],maxArtifactBytes=4_000_000,backend="hierarchy"}={}){
 if(!CLUSTERS.includes(preferred)||!["hierarchy","flat"].includes(backend)||!int(maxArtifactBytes,0,4_000_000)||!Array.isArray(disabled)||disabled.some(id=>!topology.nodes.some(n=>n.id===id)))throw Error("invalid routing policy");
 const domain=classify(question),blocked=new Set(disabled),order=[...CLUSTERS].sort((a,b)=>a===b?0:a===preferred?-1:b===preferred?1:cmp(a,b));
 let inspections=0,lookups=0;const eligible=n=>!blocked.has(n.id)&&n.bytes<=maxArtifactBytes;
 if(!domain)return {status:"abstain",domain:null,reason:"unclassified",inspections,lookups};
 if(backend==="flat"){
  const choices=topology.nodes.filter(n=>{inspections++;return n.domain===domain&&eligible(n)});
  choices.sort((a,b)=>order.indexOf(a.cluster)-order.indexOf(b.cluster));
  if(choices.length)return {status:"selected",domain,node:choices[0],inspections,lookups};
 }else{
  for(const c of order){lookups++;const n=topology.clusters.get(c).get(domain);inspections++;if(n&&eligible(n))return {status:"selected",domain,node:n,inspections,lookups}}
 }
 return {status:"abstain",domain,reason:"no eligible node",inspections,lookups};
}
export class ArtifactCache{
 constructor(topology,load,budget){if(typeof load!=="function"||!int(budget,0,144_000_000))throw Error("invalid cache policy");this.topology=topology;this.load=load;this.budget=budget;this.entries=new Map();this.residentBytes=0;this.peakBytes=0;this.loads=0;this.hits=0;this.evictions=0;this.loadedBytes=0}
 acquire(key){
  const a=this.topology.artifacts.get(key);if(!a||a.bytes>this.budget)return null;
  if(this.entries.has(key)){const model=this.entries.get(key);this.entries.delete(key);this.entries.set(key,model);this.hits++;return model}
  // Evict before loading; failed load leaves cache accounting consistent.
  while(this.residentBytes+a.bytes>this.budget){const oldest=this.entries.keys().next().value;this.entries.delete(oldest);this.residentBytes-=this.topology.artifacts.get(oldest).bytes;this.evictions++}
  const model=this.load(a);if(!model||model.bytes!==a.bytes||model.parameters!==a.parameters)throw Error("artifact metadata mismatch");
  this.entries.set(key,model);this.residentBytes+=a.bytes;this.peakBytes=Math.max(this.peakBytes,this.residentBytes);this.loads++;this.loadedBytes+=a.bytes;return model;
 }
 stats(){return {budget:this.budget,residentBytes:this.residentBytes,peakBytes:this.peakBytes,loads:this.loads,hits:this.hits,evictions:this.evictions,loadedBytes:this.loadedBytes,keys:[...this.entries.keys()]}}
}
export function queryCluster(topology,cache,question,policy={},infer=generate){
 const route=routeCluster(topology,question,{...policy,maxArtifactBytes:Math.min(policy.maxArtifactBytes??4_000_000,cache.budget)});
 if(route.status!=="selected")return {answer:null,status:"abstain",route};
 const m=cache.acquire(route.node.artifact);
 if(!m)return {answer:null,status:"abstain",route};
 const r=infer(m,question);
 return {answer:r.answer,status:"unverified",route};
}
