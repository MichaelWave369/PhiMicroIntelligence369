// Residual adapters at the exit of every transformer block.
const HASH=/^[a-f0-9]{64}$/;
export function parseLayerGear(buffer){
 const b=buffer instanceof Uint8Array?buffer:new Uint8Array(buffer),v=new DataView(b.buffer,b.byteOffset,b.byteLength);
 let p=0;const need=n=>{if(p+n>b.length)throw Error("truncated layer gear")};
 need(42);const magic=String.fromCharCode(...b.slice(0,4));if(!["LRF1","LRQ1"].includes(magic))throw Error("invalid layer gear");p=4;
 const u=()=>{const x=v.getUint16(p,true);p+=2;return x},D=u(),L=u(),R=u();
 if(D<1||D>128||L<1||L>8||R<1||R>64)throw Error("invalid dimensions");
 const baseHash=[...b.slice(p,p+32)].map(x=>x.toString(16).padStart(2,"0")).join("");p+=32;
 function f(){need(4);const x=v.getFloat32(p,true);p+=4;if(!Number.isFinite(x))throw Error("nonfinite factor");return x}
 function table(rows,cols){const w=new Float32Array(rows*cols);for(let r=0;r<rows;r++){if(magic==="LRQ1"){const s=f();if(s<=0)throw Error("invalid scale");need(cols);for(let c=0;c<cols;c++){const q=v.getInt8(p++);if(q===-128)throw Error("invalid int8");w[r*cols+c]=q*s;if(!Number.isFinite(w[r*cols+c]))throw Error("nonfinite decoded factor")}}else for(let c=0;c<cols;c++)w[r*cols+c]=f()}return w}
 const blocks=Array.from({length:L},()=>({A:table(R,D),B:table(D,R)}));if(p!==b.length)throw Error("trailing bytes");
 return {D,L,R,blocks,baseHash,format:magic==="LRQ1"?"int8":"fp32",bytes:b.length,parameters:2*D*L*R};
}
export function withLayerGears(base,entries,baseHash,{maxRank=36}={}){
 if(!HASH.test(baseHash)||base.gear||base.layerGears||!Array.isArray(entries)||entries.length<1||entries.length>9||!Number.isInteger(maxRank)||maxRank<1||maxRank>64)throw Error("invalid activation");
 const ids=new Set(),list=[];
 for(const e of entries){
  if(!e||typeof e.id!=="string"||!e.id||ids.has(e.id)||!Number.isFinite(e.weight)||e.weight<=0||e.weight>1||e.gear.baseHash!==baseHash||e.gear.D!==base.D||e.gear.L!==base.L)throw Error("incompatible activation");
  ids.add(e.id);list.push({...e});
 }
 if(list.reduce((s,e)=>s+e.gear.R,0)>maxRank)throw Error("rank budget exceeded");
 list.sort((a,b)=>a.id<b.id?-1:a.id>b.id?1:0);
 return {...base,layerGears:list};
}
export function delta(gear,layer,x){
 if(x.length!==gear.D||!Number.isInteger(layer)||layer<0||layer>=gear.L)throw Error("shape mismatch");
 const {A,B}=gear.blocks[layer],z=new Float32Array(gear.R),out=new Float32Array(gear.D);
 for(let r=0;r<gear.R;r++)for(let i=0;i<gear.D;i++)z[r]+=A[r*gear.D+i]*x[i];
 for(let j=0;j<gear.D;j++){let s=0;for(let r=0;r<gear.R;r++)s+=B[j*gear.R+r]*z[r];out[j]=s;if(!Number.isFinite(out[j]))throw Error("nonfinite residual")}
 return out;
}
export function applyLayerGears(entries,layer,x){
 const y=Float64Array.from(x);
 // All residuals see the same pre-adapter state, never sequentially chained.
 for(const e of entries){const d=delta(e.gear,layer,x);for(let i=0;i<y.length;i++)y[i]+=e.weight*d[i]}
 const out=Float32Array.from(y);if(!out.every(Number.isFinite))throw Error("nonfinite composition");return out;
}
