import {generate} from "./tiny.mjs";
export const THREE_DOMAINS=["network","science","language"];
export const DOMAINS=[...THREE_DOMAINS,"coding","geography","units","geometry","music","astronomy"];
const rules={network:/\b(?:stand for|full form|expansion|expand)\b/i,science:/\b(?:chemical symbol|element symbol)\b/i,language:/\b(?:plural|pluralize)\b/i,coding:/\bcoding acronym\b/i,geography:/\bcapital(?: city)?\b/i,units:/\bsi (?:unit )?symbol\b/i,geometry:/\b(?:polygon sides?|polygon side count|how many sides)\b/i,music:/\bsemitones?\b/i,astronomy:/\bplanet order\b/i};
export function classify(q){const hits=DOMAINS.filter(d=>rules[d].test(q)&&!(d==="network"&&rules.coding.test(q)));return hits.length===1?hits[0]:null}
export function specialist(models,q){const domain=classify(q);if(!domain||!models[domain])return {domain,status:"abstain",answer:null};return {domain,...generate(models[domain],q)}}
export function reference(trainRows,q){
 const domain=classify(q);if(!domain)return {domain,status:"abstain",answer:null};
 const tokens=" "+(q.toLowerCase().match(/[a-z0-9]+/g)||[]).join(" ")+" ";
 const matches=new Map(trainRows.filter(r=>r.domain===domain&&tokens.includes(" "+r.subject+" ")).map(r=>[r.subject,r]));
 if(matches.size!==1)return {domain,status:"abstain",answer:null};
 const r=matches.values().next().value;return {domain,status:"reference-match",answer:r.answer,source:r.source};
}
export function grade(expected,r){if(expected===null)return r.status==="abstain";return r.status!=="abstain"&&typeof r.answer==="string"&&r.answer.trim()===expected.trim()}
export function metrics(rows){
 let n=rows.length,covered=rows.filter(r=>r.status!=="abstain").length,correct=rows.filter(r=>r.correct).length,answeredCorrect=rows.filter(r=>r.status!=="abstain"&&r.correct).length,unknown=rows.filter(r=>r.expected===null);
 return {n,correct,accuracy:n?correct/n:null,coverage:n?covered/n:null,selectiveAccuracy:covered?answeredCorrect/covered:null,unknownFalseAnswerRate:unknown.length?unknown.filter(r=>r.status!=="abstain").length/unknown.length:null};
}
