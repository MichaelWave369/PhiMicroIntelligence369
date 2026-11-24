// Fixed deterministic workload, independent of model answers.
import {DOMAINS} from "./mesh.mjs";
import {pocketAddress} from "./recursive-mesh.mjs";
export function scenarios(mesh,questions){
 const qs=questions.slice(0,9);if(qs.length!==9||new Set(qs.map(q=>q.domain)).size!==9||DOMAINS.some(d=>!qs.some(q=>q.domain===d)))throw Error("one question per domain required");
 const result=[];
 for(const q of qs){
  const preferred=pocketAddress(mesh.count-1),id=preferred+"."+q.domain;
  const all=mesh.byDomain.get(q.domain).map(n=>n.id);
  for(const [kind,extra] of [
   ["healthy",{}],["slot-failure",{disabled:[id]}],["branch-failure",{blockedPrefixes:[preferred.slice(0,2)]}],
   ["all-slots-failed",{disabled:all}],["root-failure",{blockedPrefixes:[""]}],["byte-budget",{maxArtifactBytes:0}],["depth-budget",{maxDepth:3}]
  ])result.push({kind,question:q.question,domain:q.domain,expected:q.answer,policy:{preferred,...extra}});
 }
 return result;
}
