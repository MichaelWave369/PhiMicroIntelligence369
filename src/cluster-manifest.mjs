import {DOMAINS} from "./mesh.mjs";
import {CLUSTERS} from "./clusters.mjs";
export function sharedManifest(nine){
 if(nine.profile!=="nine"||!nine.models||!Array.isArray(nine.domains)||nine.domains.length!==9||DOMAINS.some(d=>!nine.domains.includes(d)))throw Error("nine-model manifest required");
 const artifacts={},slots={};
 for(const d of DOMAINS)artifacts[d]=nine.models[d];
 for(const c of CLUSTERS)for(const d of DOMAINS)slots[c+"."+d]=d;
 return {schema:1,mode:"shared-nine",artifacts,slots};
}
