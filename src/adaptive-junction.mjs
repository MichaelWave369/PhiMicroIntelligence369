// Calibration-only parent selection; no evaluation labels enter this function.
export function chooseJunction(rows,domain,candidates,loss){
 if(!Array.isArray(rows)||rows.some(r=>r.split!=="calibration")||!Array.isArray(candidates)||new Set(candidates).size!==candidates.length||!candidates.includes("none")||typeof loss!=="function")throw Error("invalid calibration");
 const sample=rows.filter(r=>r.domain===domain);if(sample.length!==6||sample.some(r=>typeof r.answer!=="string"))throw Error("six calibration questions required");
 const scores=candidates.map(id=>{const values=sample.map(q=>loss(id,q));if(values.some(x=>!Number.isFinite(x)))throw Error("nonfinite calibration loss");return {id,nll:values.reduce((s,v)=>s+v,0)/values.length}});
 scores.sort((a,b)=>a.nll-b.nll||(a.id==="none"?-1:b.id==="none"?1:a.id.localeCompare(b.id)));
 return {selected:scores[0].id,scores};
}
export function activation(variant,domain,parent){return [...(parent==="none"?[]:[variant+"-"+parent]),variant+"-c-"+domain]}
