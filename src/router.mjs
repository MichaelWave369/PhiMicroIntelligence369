// Small deterministic baseline: exact aliases only; no invented facts.
export const facts=[
 {id:"n1",domain:"network",aliases:["what is dns","dns"],answer:"DNS resolves domain names to records, including IP addresses.",source:"https://www.rfc-editor.org/rfc/rfc1034"},
 {id:"n2",domain:"network",aliases:["what is dhcp","dhcp"],answer:"DHCP supplies configuration parameters to network hosts.",source:"https://www.rfc-editor.org/rfc/rfc2131"},
 {id:"s1",domain:"science",aliases:["what is the symbol for gold","symbol for gold"],answer:"Au",source:"https://iupac.org/what-we-do/periodic-table-of-elements/"},
 {id:"s2",domain:"science",aliases:["what is the symbol for carbon","symbol for carbon"],answer:"C",source:"https://iupac.org/what-we-do/periodic-table-of-elements/"},
 {id:"c1",domain:"coding",aliases:["what is json","json"],answer:"JSON is a text format for structured data interchange.",source:"https://www.rfc-editor.org/rfc/rfc8259"},
 {id:"l1",domain:"language",aliases:["plural of mouse","what is the plural of mouse"],answer:"mice",source:"https://dictionary.cambridge.org/dictionary/english/mouse"}
];
export const normalize=s=>s.toLowerCase().trim().replace(/[?.!]+$/g,"").replace(/\s+/g," ");
export function route(question){
 const q=normalize(question),match=facts.filter(f=>f.aliases.includes(q));
 if(match.length===1){const f=match[0];return {status:"reference-match",domain:f.domain,answer:f.answer,source:f.source,id:f.id}}
 const a=q.match(/^(?:what is )?(-?\d{1,6})\s*([+*-])\s*(-?\d{1,6})$/);
 if(a){const x=Number(a[1]),y=Number(a[3]);return {status:"computed",domain:"logic",answer:String(a[2]==="+"?x+y:a[2]==="-"?x-y:x*y),source:"Exact bounded integer arithmetic"}}
 return {status:"abstain",domain:"unknown",answer:"No matching reference or supported operation.",source:null}
}
export const evaluation=[
 ["what is dns","DNS resolves domain names to records, including IP addresses."],["DHCP?","DHCP supplies configuration parameters to network hosts."],["what is the symbol for gold","Au"],["plural of mouse","mice"],["12 * 7","84"],["-3 + 5","2"],["what is blorbium",null],["solve all of physics",null]
];
export function benchmark(){return evaluation.map(([q,expected])=>{const r=route(q);return {question:q,expected,result:r.answer,status:r.status,pass:expected===null?r.status==="abstain":r.answer===expected}})}
