import test from "node:test";import assert from "node:assert/strict";
import {parseTiny,generate,encode,TinySession} from "../src/tiny.mjs";
import {route,benchmark} from "../src/router.mjs";
function fixture(){const D=4,L=1,H=2,M=6,a=[];const b=new ArrayBuffer(4);const v=new DataView(b);function f(x){v.setFloat32(0,x,true);a.push(...new Uint8Array(b))}function t(rows,cols){for(let i=0;i<rows;i++){f(1);for(let j=0;j<cols;j++)a.push(0)}}function g(){for(let i=0;i<D;i++)f(1)}a.push(84,78,89,50,D,0,L,0,H,0,M,0);t(97,D);t(128,D);g();for(let i=0;i<4;i++)t(D,D);g();t(M,D);t(D,M);g();return Uint8Array.from(a)}
test("TNY2 parsing and dimensions",()=>{const m=parseTiny(fixture());assert.equal(m.D,4);assert.equal(m.H,2);assert.equal(m.parameters,1024);assert.equal(m.bytes,fixture().length)});
test("truncation and magic rejected",()=>{assert.throws(()=>parseTiny(fixture().slice(0,-1)));assert.throws(()=>parseTiny(new Uint8Array(12)))});
test("invalid dimensions rejected",()=>{const f=fixture();f[8]=3;assert.throws(()=>parseTiny(f))});
test("zero-weight forward pass returns finite zero logits",()=>{const s=new TinySession(parseTiny(fixture()));assert.deepEqual([...s.step(2)],Array(97).fill(0));assert.equal(s.p,1);assert.deepEqual([...s.step(1)],Array(97).fill(0));assert.equal(s.p,2)});
test("greedy generation stops on padding",()=>{assert.equal(generate(parseTiny(fixture()),"hi").answer,"");assert.throws(()=>encode("π"));assert.throws(()=>generate(parseTiny(fixture()),"x".repeat(73)))});
test("baseline arithmetic and abstention",()=>{assert.equal(route("12 * 7").answer,"84");assert.equal(route("unknown").status,"abstain");assert.equal(route("what is DNS?").domain,"network");assert.ok(benchmark().every(r=>r.pass))});
