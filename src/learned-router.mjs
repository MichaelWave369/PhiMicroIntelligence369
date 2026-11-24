import {DOMAINS} from "./mesh.mjs";
const grams=s=>{const t="^"+s.toLowerCase().replace(/[^a-z0-9 ]/g," ").replace(/\s+/g," ").trim()+"$";const a=[];for(let i=0;i+2<t.length;i++)a.push(t.slice(i,i+3));return a};
export function fitRouter(rows){
 const counts=new Map(DOMAINS.map(d=>[d,new Map()])),totals=new Map(DOMAINS.map(d=>[d,0])),vocab=new Set();
 for(const r of rows){if(r.split!=="train"||!counts.has(r.domain))throw Error("router training rows only");const m=counts.get(r.domain);for(const g of grams(r.question)){vocab.add(g);m.set(g,(m.get(g)||0)+1);totals.set(r.domain,totals.get(r.domain)+1)}}
 if(DOMAINS.some(d=>!totals.get(d)))throw Error("missing training domain");
 return {counts,totals,vocab};
}
export function rankDomains(model,q){
 const features=grams(q).filter(g=>model.vocab.has(g));if(!features.length)return [];
 return DOMAINS.map(domain=>({domain,score:features.reduce((s,g)=>s+Math.log(((model.counts.get(domain).get(g)||0)+1)/(model.totals.get(domain)+model.vocab.size)),0)})).sort((a,b)=>b.score-a.score||DOMAINS.indexOf(a.domain)-DOMAINS.indexOf(b.domain));
}
