import {DOMAINS} from "./mesh.mjs";
import {withLayerGears} from "./layer-gears.mjs";
export function validateGroups(groups){
 if(!groups||Object.keys(groups).length!==3||Object.values(groups).some(a=>!Array.isArray(a)||a.length!==3))throw Error("three groups of three required");
 const ds=Object.values(groups).flat();if(new Set(ds).size!==9||DOMAINS.some(d=>!ds.includes(d)))throw Error("domain partition mismatch");
 return Object.fromEntries(Object.entries(groups).flatMap(([p,ds])=>ds.map(d=>[d,p])));
}
export function validateHierarchy(manifest,groups){
 const parents=validateGroups(groups);
 if(manifest?.schema!==1||manifest.kind!=="nested-layer-residual"||JSON.stringify(manifest.groups)!==JSON.stringify(groups)||!manifest.gears)throw Error("invalid hierarchy");
 const keys=[...Object.keys(groups).map(p=>"parent-"+p),...DOMAINS.flatMap(d=>["child-"+d,"flat-"+d,"active-"+d]),"pooled"];
 if(Object.keys(manifest.gears).length!==keys.length||keys.some(k=>!manifest.gears[k]))throw Error("gear inventory mismatch");
 for(const d of DOMAINS){
  const p=manifest.gears["parent-"+parents[d]],c=manifest.gears["child-"+d];
  if(c.parent!=="parent-"+parents[d]||c.parentSha256!==p.fp32.sha256)throw Error("child parent binding mismatch");
 }
 for(const k of keys){const g=manifest.gears[k],rank=k==="pooled"?36:k.startsWith("flat-")?4:k.startsWith("active-")?6:3;
  if(g.rank!==rank||g.fp32.parameters!==128*rank||g.int8.parameters!==128*rank)throw Error("gear rank mismatch "+k)}
 const sum=ids=>ids.reduce((s,id)=>s+manifest.gears[id].fp32.parameters,0),samples=ids=>ids.reduce((s,id)=>s+manifest.gears[id].examples,0);
 const nested=[...Object.keys(groups).map(p=>"parent-"+p),...DOMAINS.map(d=>"child-"+d)],flat=DOMAINS.map(d=>"flat-"+d),active=DOMAINS.map(d=>"active-"+d);
 if(sum(nested)!==sum(flat)||sum(flat)!==sum(["pooled"])||samples(nested)!==samples(flat)||samples(flat)!==samples(["pooled"])||samples(active)!==samples(flat))throw Error("unmatched budget");
 return {parents,keys,parameters:{nested:sum(nested),flat:sum(flat),pooled:sum(["pooled"]),active:sum(active)},examples:samples(nested)};
}
export function selectNested(base,registry,manifest,domain,{format="int8",mode="nested",maxRank=6}={}){
 const parents=validateGroups(manifest.groups);if(!parents[domain])throw Error("unknown domain");
 const parent="parent-"+parents[domain],child="child-"+domain;
 let ids;
 switch(mode){case "nested":ids=[parent,child];break;case "parent":ids=[parent];break;case "child":ids=[child];break;case "wrong-parent":ids=["parent-"+Object.keys(manifest.groups).find(p=>p!==parents[domain]),child];break;case "flat":ids=["flat-"+domain];break;case "active":ids=["active-"+domain];break;case "pooled":ids=["pooled"];break;default:throw Error("unknown mode")}
 return {ids,model:withLayerGears(base,ids.map(id=>({id,weight:1,gear:registry[id][format]})),manifest.base.sha256,{maxRank})};
}
const features=q=>{const s="^"+q.toLowerCase().replace(/[^a-z0-9 ]/g," ").replace(/\s+/g," ").trim()+"$";return Array.from({length:Math.max(0,s.length-2)},(_,i)=>s.slice(i,i+3))};
function fit(rows,labels,labelOf){
 const vocab=new Set(),counts=new Map(labels.map(k=>[k,new Map()])),totals=new Map(labels.map(k=>[k,0]));
 for(const r of rows){const k=labelOf(r),m=counts.get(k);if(!m)throw Error("unknown label");for(const f of features(r.question)){vocab.add(f);m.set(f,(m.get(f)||0)+1);totals.set(k,totals.get(k)+1)}}
 if(labels.some(k=>!totals.get(k)))throw Error("missing label");return {labels,vocab,counts,totals};
}
function pick(m,q){const fs=features(q).filter(f=>m.vocab.has(f));if(!fs.length)return null;return m.labels.map(k=>({k,s:fs.reduce((s,f)=>s+Math.log(((m.counts.get(k).get(f)||0)+1)/(m.totals.get(k)+m.vocab.size)),0)})).sort((a,b)=>b.s-a.s||m.labels.indexOf(a.k)-m.labels.indexOf(b.k))[0].k}
export function fitNestedRouter(rows,groups){
 const map=validateGroups(groups);if(rows.some(r=>r.split!=="train"||!map[r.domain]))throw Error("training rows only");
 return {parent:fit(rows,Object.keys(groups),r=>map[r.domain]),children:Object.fromEntries(Object.entries(groups).map(([p,ds])=>[p,fit(rows.filter(r=>map[r.domain]===p),ds,r=>r.domain)]))};
}
export function routeNested(model,question){const parent=pick(model.parent,question);return {parent,domain:parent?pick(model.children[parent],question):null}}
