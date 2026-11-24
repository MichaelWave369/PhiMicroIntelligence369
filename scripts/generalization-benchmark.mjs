import {readFileSync,writeFileSync} from "node:fs";import {join} from "node:path";import {spawnSync} from "node:child_process";import {fileURLToPath} from "node:url";
const root=process.argv[2],systems=["reference","normalized-reference","raw32","aug32","aug64","canonical32"];if(!root)throw Error("run directory required");
const results={};for(const system of systems){const p=spawnSync(process.execPath,[fileURLToPath(new URL("./generalization-worker.mjs",import.meta.url)),root,system],{stdio:"inherit"});if(p.status!==0)throw Error("worker failed "+system);results[system]=JSON.parse(readFileSync(join(root,system+".report.json"),"utf8"))}
const seed=results.raw32.seed;writeFileSync(join(root,"generalization-benchmark.json"),JSON.stringify({schema:1,seed,systems:results},null,2)+"\n");
