import {readFileSync,writeFileSync} from "node:fs";import {join} from "node:path";
const [out,...roots]=process.argv.slice(2);if(!out||roots.length!==3)throw Error("provide output and three seed directories");
const reports=roots.map(r=>JSON.parse(readFileSync(join(r,"nested-benchmark.json"),"utf8")));if(reports.map(r=>r.seed).sort((a,b)=>a-b).join(",")!=="369,636,963")throw Error("preregistered seeds required");
const keys=Object.keys(reports[0].summary);if(reports.some(r=>JSON.stringify(Object.keys(r.summary))!==JSON.stringify(keys)||r.adaptationExamples!==reports[0].adaptationExamples||JSON.stringify(r.budgets)!==JSON.stringify(reports[0].budgets)))throw Error("incompatible reports");
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const summary=Object.fromEntries(keys.map(k=>{const a=reports.map(r=>r.summary[k].accuracy),n=reports.map(r=>r.summary[k].meanNll);return [k,{accuracy:a,meanAccuracy:mean(a),minAccuracy:Math.min(...a),maxAccuracy:Math.max(...a),nll:n,meanNll:n.every(Number.isFinite)?mean(n):null}]}));
const result={schema:1,seeds:reports.map(r=>r.seed),budgets:reports[0].budgets,examples:reports[0].adaptationExamples,summary,contrasts:reports.map(r=>({seed:r.seed,contrasts:r.contrasts}))};writeFileSync(out,JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify(result,null,2));
