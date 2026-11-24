// Subject extraction uses only known training subjects, never their answers.
import {classify} from "./mesh.mjs";
import {fitRouter,rankDomains} from "./learned-router.mjs";
const tokens=s=>" "+(s.toLowerCase().match(/[a-z0-9]+/g)||[]).join(" ")+" ";
export function createNormalizer(rows){
 if(!Array.isArray(rows)||rows.some(r=>r.split!=="train"))throw Error("training only");
 const items=[...new Map(rows.map(r=>[r.domain+":"+r.subject,{domain:r.domain,subject:r.subject}])).values()];
 return {items,router:fitRouter(rows)};
}
export function normalizeQuestion(normalizer,q){
 const text=tokens(q),lex=classify(q),learned=rankDomains(normalizer.router,q)[0]?.domain??null;
 const domain=lex??learned;if(!domain)return {status:"abstain",reason:"no domain",domain:null};
 const hits=normalizer.items.filter(r=>r.domain===domain&&text.includes(tokens(r.subject)));
 if(hits.length!==1)return {status:"abstain",reason:"no unique known subject",domain};
 return {status:"matched",domain,subject:hits[0].subject,question:domain+": "+hits[0].subject,route:lex?"lexical":"learned"};
}
