// Local, user-asserted evidence. Never promotes neural text automatically.
export const MEMORY_KEY="phi-micro-evidence-v1";
const MAX=256;
export const questionKey=q=>String(q).trim().toLowerCase().replace(/[?.!]+$/,"").replace(/\s+/g," ");
function valid(e){return e&&typeof e.question==="string"&&questionKey(e.question).length>0&&e.question.length<=256&&typeof e.answer==="string"&&e.answer.trim().length>0&&e.answer.length<=512&&typeof e.source==="string"&&e.source.trim().length>0&&e.source.length<=512&&typeof e.createdAt==="string"&&Number.isFinite(Date.parse(e.createdAt))}
export class EvidenceMemory{
 constructor(storage){this.storage=storage;this.rows=[];const raw=storage.getItem(MEMORY_KEY);if(raw!==null){const data=JSON.parse(raw);if(!Array.isArray(data)||data.length>MAX||!data.every(valid))throw Error("Invalid saved evidence");this.rows=data}}
 add(question,answer,source){const e={question:question.trim(),answer:answer.trim(),source:source.trim(),createdAt:new Date().toISOString()};if(!valid(e))throw Error("Question, answer and source are required within size limits");if(this.rows.length>=MAX)throw Error("Evidence limit reached");const next=[...this.rows,e];this.storage.setItem(MEMORY_KEY,JSON.stringify(next));this.rows=next;return e}
 lookup(q){return this.rows.filter(e=>questionKey(e.question)===questionKey(q)).map(e=>({...e}))}
 clear(){this.storage.removeItem(MEMORY_KEY);this.rows=[]}
}
// Each candidate identifies its origin. Agreement is not independent verification.
export function arbitrate(question,candidates,evidence=[]){
 const usable=candidates.filter(c=>typeof c.answer==="string"&&c.answer.trim()&&c.status!=="abstain");
 const claims=evidence.filter(e=>valid(e)&&questionKey(e.question)===questionKey(question));
 const ev=new Set(claims.map(e=>e.answer.trim()));
 const neural=new Set(usable.map(c=>c.answer.trim()));
 if(ev.size>1)return {status:"evidence-conflict",answer:null,candidates:usable,evidence:claims};
 if(ev.size===1){const answer=[...ev][0];return {status:neural.size&&[...neural].some(x=>x!==answer)?"evidence-model-disagreement":"user-evidence",answer,candidates:usable,evidence:claims}}
 if(neural.size>1)return {status:"model-disagreement",answer:null,candidates:usable,evidence:[]};
 if(neural.size===1)return {status:new Set(usable.map(c=>c.origin)).size>1?"unverified-agreement":"unverified-single",answer:[...neural][0],candidates:usable,evidence:[]};
 return {status:"abstain",answer:null,candidates:usable,evidence:[]};
}
