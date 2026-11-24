// Local bounded lifecycle and delegation; no network or autonomous authority.
import {DOMAINS,classify} from "./mesh.mjs";
import {buildMesh,pocketAddress,routePocket,MAX_POCKETS} from "./recursive-mesh.mjs";
import {generate} from "./tiny.mjs";
const copy=x=>JSON.parse(JSON.stringify(x));
const int=(x,min,max)=>Number.isSafeInteger(x)&&x>=min&&x<=max;
const validPrefix=p=>typeof p==="string"&&/^[012]{0,4}$/.test(p);
const addresses=Array.from({length:MAX_POCKETS},(_,i)=>pocketAddress(i));
function fingerprint(mesh){return JSON.stringify([...mesh.artifacts.values()].map(a=>[a.key,a.sha256,a.bytes,a.parameters]))}
function domainsValid(ds){return Array.isArray(ds)&&ds.length>0&&ds.length<=DOMAINS.length&&new Set(ds).size===ds.length&&ds.every(d=>DOMAINS.includes(d))}
function validateState(s,key,maxActive){
 if(!s||s.schema!==1||s.artifacts!==key||s.maxActive!==maxActive||!int(s.revision,0,Number.MAX_SAFE_INTEGER)||!Array.isArray(s.pockets)||s.pockets.length>MAX_POCKETS)throw Error("invalid snapshot or artifact drift");
 const seen=new Set();let active=0;
 for(const p of s.pockets){
  if(!p||!addresses.includes(p.address)||p.id!=="pocket:"+p.address||seen.has(p.address)||!["active","retired"].includes(p.status)||!int(p.generation,1,s.revision)||!domainsValid(p.domains)||!int(p.createdRevision,1,s.revision)||!int(p.changedRevision,p.createdRevision,s.revision))throw Error("invalid pocket record");
  seen.add(p.address);if(p.status==="active")active++;
 }
 if(active>maxActive)throw Error("active budget exceeded");
}
export class ElasticMesh{
 constructor(manifest,store,{maxActive=41}={}){
  if(!int(maxActive,1,MAX_POCKETS)||typeof store?.read!=="function"||typeof store?.write!=="function")throw Error("invalid lifecycle policy");
  this.mesh=buildMesh(manifest,MAX_POCKETS);this.store=store;this.key=fingerprint(this.mesh);this.maxActive=maxActive;
  const existing=store.read();this.state=existing===null?{schema:1,artifacts:this.key,maxActive,revision:0,pockets:[]}:copy(existing);
  validateState(this.state,this.key,maxActive);
 }
 snapshot(){return copy(this.state)}
 commit(next){validateState(next,this.key,this.maxActive);this.store.write(copy(next),this.state.revision);this.state=next;return this.snapshot()}
 create({address,domains=DOMAINS}={}){
  if(!domainsValid(domains))throw Error("invalid capabilities");
  const next=this.snapshot();
  if(next.pockets.filter(p=>p.status==="active").length>=this.maxActive)throw Error("active pocket budget");
  const p=address===undefined?addresses.find(a=>!next.pockets.some(p=>p.address===a)):address;
  if(!addresses.includes(p)||next.pockets.some(x=>x.address===p))throw Error("address unavailable; retired identities cannot be reassigned");
  next.revision++;next.pockets.push({id:"pocket:"+p,address:p,status:"active",generation:1,domains:[...domains].sort(),createdRevision:next.revision,changedRevision:next.revision});return this.commit(next);
 }
 retire(address){return this.change(address,"retired")}
 restore(address){return this.change(address,"active")}
 change(address,status){
  const next=this.snapshot(),p=next.pockets.find(p=>p.address===address);
  if(!p||p.status===status)throw Error("invalid lifecycle transition");
  if(status==="active"&&next.pockets.filter(p=>p.status==="active").length>=this.maxActive)throw Error("active pocket budget");
  next.revision++;p.status=status;p.changedRevision=next.revision;if(status==="active")p.generation++;return this.commit(next);
 }
 discover(domain,{scope="",blockedPrefixes=[]}={}){
  if(!DOMAINS.includes(domain)||!validPrefix(scope)||!Array.isArray(blockedPrefixes)||blockedPrefixes.length>121||blockedPrefixes.some(p=>!validPrefix(p)))throw Error("invalid discovery policy");
  return this.state.pockets.filter(p=>p.status==="active"&&p.domains.includes(domain)&&p.address.startsWith(scope)&&!blockedPrefixes.some(x=>p.address.startsWith(x))).sort((a,b)=>a.address.localeCompare(b.address)).map(p=>({id:p.id,address:p.address,generation:p.generation,domain,artifact:domain}));
 }
 delegate(question,cache,{preferred="0000",scope="",blockedPrefixes=[],fuel=81,maxArtifactBytes=4_000_000}={},infer=generate){
  if(typeof question!=="string"||question.length>72||!/^[ -~]+$/.test(question)||!addresses.includes(preferred)||!int(fuel,0,81)||!int(maxArtifactBytes,0,4_000_000)||!validPrefix(scope)||!Array.isArray(blockedPrefixes)||blockedPrefixes.length>121||blockedPrefixes.some(p=>!validPrefix(p))||typeof cache?.acquire!=="function"||!int(cache.budget,0,144_000_000))throw Error("invalid delegation");
  const domain=classify(question),revision=this.state.revision,events=[];
  const finish=(reason,answer=null,pocket=null)=>({schema:1,revision,domain,scope,status:answer===null?"abstain":"unverified",reason,answer,pocket,fuelUsed:events.length,events});
  if(!domain)return finish("unclassified");
  const artifact=this.mesh.artifacts.get(domain);if(artifact.bytes>Math.min(cache.budget,maxArtifactBytes))return finish("artifact-budget");
  const choices=this.discover(domain,{scope,blockedPrefixes}).sort((a,b)=>a.address===b.address?0:a.address===preferred?-1:b.address===preferred?1:a.address.localeCompare(b.address));
  if(!choices.length)return finish("no capability in scope");
  if(fuel===0)return finish("fuel exhausted");
  // One local delegation only. No autonomous recursive retries; exceptions fail closed.
  const p=choices[0];events.push({seq:1,type:"delegate",from:"root",to:p.id,generation:p.generation,artifact:domain,sha256:artifact.sha256});
  try{
   const model=cache.acquire(domain);if(!model)return finish("cache refused");
   const r=infer(model,question);if(!r||typeof r.answer!=="string"||!r.answer.trim()||r.answer.length>512)return finish("empty or invalid answer");
   return finish("local inference",r.answer,p);
  }catch(e){return finish("execution error")}
 }
}
// Compare-and-swap store for tests and single-process browser adapters.
export function memoryStore(initial=null){
 let value=copy(initial);
 return {read:()=>copy(value),write:(next,expected)=>{if((value?.revision??0)!==expected)throw Error("stale revision");value=copy(next)}};
}
