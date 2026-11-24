import {readFileSync} from "node:fs";import {join} from "node:path";
const roots=process.argv.slice(2);if(roots.length<3)throw Error("at least three independent seeds required");
const reports=roots.map(r=>JSON.parse(readFileSync(join(r,"layer-benchmark.json"),"utf8")));if(new Set(reports.map(r=>r.seed)).size!==reports.length)throw Error("duplicate seeds");
const keys=Object.keys(reports[0].summary),out={};for(const k of keys){const values=reports.map(r=>r.summary[k].accuracy),mean=values.reduce((s,v)=>s+v,0)/values.length;out[k]={values,mean,min:Math.min(...values),max:Math.max(...values)}}
console.log(JSON.stringify({seeds:reports.map(r=>r.seed),accuracy:out,interference:reports.map(r=>r.interference)},null,2));
