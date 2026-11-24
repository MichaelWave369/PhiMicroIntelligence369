// Independent JavaScript implementation of the documented TNY2 format.
// Upstream: WollyMonjac/Tiny-2.1 (MIT); see docs/UPSTREAM.md.
export const VOCAB=97, CTX=128;
export function parseTiny(buffer){
  const b=buffer instanceof Uint8Array?buffer:new Uint8Array(buffer);
  const v=new DataView(b.buffer,b.byteOffset,b.byteLength);let p=0;
  function need(n){if(p+n>b.length)throw Error("Truncated TNY2 file");}
  need(12);if(String.fromCharCode(...b.slice(0,4))!=="TNY2")throw Error("Expected TNY2 magic");
  p=4;const u=()=>{need(2);const x=v.getUint16(p,true);p+=2;return x};
  const D=u(),L=u(),H=u(),M=u();
  if(D<1||D>128||L<1||L>8||H<1||D%H||M<1||M>512)throw Error("Invalid model dimensions");
  function f(){need(4);const x=v.getFloat32(p,true);p+=4;if(!Number.isFinite(x))throw Error("Nonfinite weight");return x}
  function table(rows,cols){const scales=new Float32Array(rows),w=new Int8Array(rows*cols);for(let r=0;r<rows;r++){scales[r]=f();if(scales[r]<=0)throw Error("Invalid scale");need(cols);w.set(new Int8Array(b.buffer,b.byteOffset+p,cols),r*cols);p+=cols}return {rows,cols,scales,w}}
  function gains(){return Float32Array.from({length:D},f)}
  const emb=table(VOCAB,D),pos=table(CTX,D),blocks=[];
  for(let l=0;l<L;l++)blocks.push({n1:gains(),q:table(D,D),k:table(D,D),v:table(D,D),o:table(D,D),n2:gains(),f1:table(M,D),f2:table(D,M)});
  const nf=gains();if(p!==b.length)throw Error("Unexpected trailing bytes");
  return {D,L,H,M,emb,pos,blocks,nf,bytes:b.length,parameters:(VOCAB+CTX)*D+L*(4*D*D+2*M*D+2*D)+D};
}
function mv(t,x){const y=new Float32Array(t.rows);for(let r=0;r<t.rows;r++){let s=0;for(let c=0;c<t.cols;c++)s+=t.w[r*t.cols+c]*x[c];y[r]=s*t.scales[r]}return y}
function norm(x,g){let s=0;for(const z of x)s+=z*z;const k=1/Math.sqrt(s/x.length+1e-5);return Float32Array.from(x,(z,i)=>z*k*g[i])}
function quant(x){let big=1e-20;for(const z of x)big=Math.max(big,Math.abs(z));const s=big/127;return {w:Int8Array.from(x,z=>Math.max(-127,Math.min(127,Math.sign(z)*Math.round(Math.abs(z)/s)))),s}}
export function encode(s){return Array.from(s,c=>{const n=c.charCodeAt(0);if(c==="\n")return 1;if(n<32||n>126||c.length!==1)throw Error("Use printable ASCII only");return n-30})}
export class TinySession{
  constructor(model){this.m=model;this.p=0;this.cache=model.blocks.map(()=>({k:[],v:[]}));this.activity=[]}
  step(tok){const {D,H,emb,pos,blocks,nf}=this.m,p=this.p;if(p>=CTX||!Number.isInteger(tok)||tok<0||tok>=VOCAB)throw Error("Invalid token/context");
    const x=Float32Array.from({length:D},(_,i)=>emb.w[tok*D+i]*emb.scales[tok]+pos.w[p*D+i]*pos.scales[p]),hd=D/H;
    this.activity=[];
    blocks.forEach((b,l)=>{const z=norm(x,b.n1),q=mv(b.q,z),cache=this.cache[l];cache.k.push(quant(mv(b.k,z)));cache.v.push(quant(mv(b.v,z)));const a=new Float32Array(D);
      for(let h=0;h<H;h++){const off=h*hd,scores=cache.k.map(k=>{let s=0;for(let i=0;i<hd;i++)s+=q[off+i]*k.w[off+i];return s*k.s/Math.sqrt(hd)});const max=Math.max(...scores),e=scores.map(s=>Math.exp(s-max)),sum=e.reduce((a,b)=>a+b,0);
        cache.v.forEach((v,j)=>{for(let i=0;i<hd;i++)a[off+i]+=e[j]/sum*v.s*v.w[off+i]})}
      const att=mv(b.o,a);for(let i=0;i<D;i++)x[i]+=att[i];
      const mid=mv(b.f1,norm(x,b.n2));let act=0;for(let i=0;i<mid.length;i++){mid[i]=Math.max(0,mid[i]);act+=mid[i]}this.activity.push(act/mid.length);const ff=mv(b.f2,mid);for(let i=0;i<D;i++)x[i]+=ff[i]
    });this.p++;return mv(emb,norm(x,nf))}
}
export function generate(model,question,max=55,onToken=()=>{}){
  if(!Number.isInteger(max)||max<1||max>127)throw Error("Invalid generation limit");
  const ids=encode(question.trim());if(!ids.length||ids.length>72||ids.includes(1))throw Error("Question must contain 1–72 printable ASCII characters");
  const s=new TinySession(model);let logits;for(const t of [...ids,1])logits=s.step(t);let answer="",trace=[];
  while(s.p<=CTX&&answer.length<max){let best=0;for(let i=1;i<logits.length;i++)if(logits[i]>logits[best])best=i;if(best<=1)break;const z=logits[best],prob=1/logits.reduce((a,b)=>a+Math.exp(b-z),0);if(!Number.isFinite(prob))throw Error("Nonfinite logits");const c=String.fromCharCode(best+30);answer+=c;trace.push({character:c,probability:prob,activity:[...s.activity]});onToken(trace.at(-1));if(s.p===CTX)break;logits=s.step(best)}
  return {answer,trace,status:"unverified",contextUsed:s.p};
}
