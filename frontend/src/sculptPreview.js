// Preview is disposable; server local-band solve and gates decide the committed geometry.
export function adjacency(faces,n){const sets=Array.from({length:n},()=>new Set());for(const [a,b,c] of faces){sets[a].add(b).add(c);sets[b].add(a).add(c);sets[c].add(a).add(b)}return sets.map(s=>[...s])}
export function growMasks(faces,weights,locks,fixedWeights=null){const masks=weights.map((w,i)=>w>.5&&locks[i]>0?'MOVING':'FIXED'),adj=adjacency(faces,weights.length);let front=new Set(masks.flatMap((m,i)=>m==='MOVING'?[i]:[])),seen=new Set(front);for(let k=0;k<3;k++){const next=new Set();for(const i of front)for(const j of adj[i])if(!seen.has(j)){next.add(j);seen.add(j);masks[j]='ELASTIC'}front=next}return masks.map((m,i)=>locks[i]===0||fixedWeights?.[i]>.5?'FIXED':m)}
export function localBrush(base,normals,faces,centers,{radius,strength,falloff=2,operation},locks,weights=null,cachedAdj=null){
  const brush=base.map((p,i)=>Math.max(0,1-Math.min(...centers.map(c=>Math.hypot(...p.map((v,j)=>v-c[j]))))/radius)**falloff*locks[i]*(weights?.[i]??1));
  const result=base.map(p=>[...p]),adj=operation==='smooth'?(cachedAdj||adjacency(faces,base.length)):null;
  const ids=brush.flatMap((w,i)=>w>0?[i]:[]);if(!ids.length)return result;
  let normal=null,center=null;
  if(operation==='flatten'){const sum=ids.reduce((s,i)=>s+brush[i],0);center=[0,1,2].map(k=>ids.reduce((s,i)=>s+base[i][k]*brush[i],0)/sum);normal=[0,1,2].map(k=>ids.reduce((s,i)=>s+normals[i][k]*brush[i],0));const length=Math.hypot(...normal);normal=normal.map(v=>v/(length||1))}
  for(const i of ids){for(let k=0;k<3;k++){
    if(operation==='smooth'){const mean=adj[i].reduce((s,j)=>s+base[j][k],0)/(adj[i].length||1);result[i][k]+=(mean-base[i][k])*Math.min(strength*5,1)*brush[i]}
    else if(operation==='flatten'){const distance=normal.reduce((s,v,j)=>s+v*(base[i][j]-center[j]),0);result[i][k]-=normal[k]*Math.max(-strength,Math.min(strength,distance))*brush[i]}
    else result[i][k]+=normals[i][k]*strength*(operation==='remove'?-1:1)*brush[i];
  }}return result;
}

export function localDragPreview(base,centers,delta,{radius,falloff=2},locks,weights=null){
  const origin=centers?.[0]; if(!origin)return base.map(p=>[...p]);
  const length=Math.hypot(...delta); const safe=length>1?delta.map(v=>v/length):delta;
  return base.map((p,i)=>{
    const d=Math.hypot(...p.map((v,j)=>v-origin[j]));
    const w=Math.max(0,1-d/radius)**falloff*locks[i]*(weights?.[i]??1);
    return p.map((v,k)=>v+safe[k]*w);
  });
}
export function transformPreview(base,faces,desired,masks,locks){
  const out=base.map((p,i)=>masks?masks[i]==='MOVING'?p.map((v,k)=>v+(desired[i][k]-v)*locks[i]):[...p]:p.map((v,k)=>v+(desired[i][k]-v)*locks[i]));
  if(masks){const adj=adjacency(faces,base.length);let delta=out.map((p,i)=>p.map((v,k)=>v-base[i][k]));for(let t=0;t<25;t++){delta=delta.map((d,i)=>masks[i]==='ELASTIC'&&locks[i]>0?[0,1,2].map(k=>adj[i].reduce((s,j)=>s+delta[j][k],0)/(adj[i].length||1)):d)}for(let i=0;i<out.length;i++)if(masks[i]==='ELASTIC'&&locks[i]>0)out[i]=base[i].map((v,k)=>v+delta[i][k])}return out;
}
export { contactColor } from './contactMap.js';