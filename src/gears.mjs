// Output-head residual gear. Independent binary format; never mutates base weights.
const HEX=/^[a-f0-9]{64}$/;
export function parseGear(buffer){
 const b=buffer instanceof Uint8Array?buffer:new Uint8Array(buffer),v=new DataView(b.buffer,b.byteOffset,b.byteLength);let p=0;
 const need=n=>{if(p+n>b.length)throw Error("truncated gear")};
 need(42);const magic=String.fromCharCode(...b.slice(0,4));if(!["GRF1","GRQ1"].includes(magic))throw Error("invalid gear magic");p=4;
 const u=()=>{const x=v.getUint16(p,true);p+=2;return x};
 const D=u(),V=u(),R=u();if(D<1||D>128||V!==97||R<1||R>64)throw Error("invalid gear dimensions");
 const baseHash=[...b.slice(p,p+32)].map(x=>x.toString(16).padStart(2,"0")).join("");p+=32;
 function f(){need(4);const x=v.getFloat32(p,true);p+=4;if(!Number.isFinite(x))throw Error("nonfinite gear");return x}
 function table(rows,cols){
  const w=new Float32Array(rows*cols);
  for(let r=0;r<rows;r++){
   if(magic==="GRQ1"){const s=f();if(s<=0)throw Error("invalid scale");need(cols);for(let c=0;c<cols;c++){const x=v.getInt8(p++);if(x===-128)throw Error("invalid quantized weight");w[r*cols+c]=s*x;if(!Number.isFinite(w[r*cols+c]))throw Error("nonfinite decoded gear")}}
   else for(let c=0;c<cols;c++)w[r*cols+c]=f();
  }return w
 }
 const A=table(R,D),B=table(V,R);if(p!==b.length)throw Error("trailing gear data");
 return {D,V,R,A,B,baseHash,format:magic==="GRQ1"?"int8":"fp32",bytes:b.length,parameters:R*(D+V)};
}
export function withGear(base,gear,baseHash){
 if(!HEX.test(baseHash)||gear.baseHash!==baseHash||base.D!==gear.D||gear.V!==97)throw Error("gear/base mismatch");
 if(base.gear)throw Error("base already has a gear");
 return {...base,gear};
}
export function applyGear(gear,h,logits){
 if(h.length!==gear.D||logits.length!==gear.V)throw Error("gear activation shape mismatch");
 const z=new Float32Array(gear.R);
 for(let r=0;r<gear.R;r++){let x=0;for(let i=0;i<gear.D;i++)x+=gear.A[r*gear.D+i]*h[i];z[r]=x}
 const out=new Float32Array(logits);
 for(let j=0;j<gear.V;j++){let x=0;for(let r=0;r<gear.R;r++)x+=gear.B[j*gear.R+r]*z[r];out[j]+=x;if(!Number.isFinite(out[j]))throw Error("nonfinite adapted logits")}
 return out;
}
