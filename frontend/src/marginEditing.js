import {Triangle,Vector3,CatmullRomCurve3} from 'three';

export function surfaceProjector(geometry){
  const position=geometry.attributes.position,index=geometry.index,tri=new Triangle(),target=new Vector3(),hit=new Vector3();
  return point=>{
    target.fromArray(point);let best=Infinity,result=point.slice();
    const count=index?index.count:position.count;
    for(let i=0;i<count;i+=3){
      tri.a.fromBufferAttribute(position,index?index.getX(i):i);
      tri.b.fromBufferAttribute(position,index?index.getX(i+1):i+1);
      tri.c.fromBufferAttribute(position,index?index.getX(i+2):i+2);
      tri.closestPointToPoint(target,hit);const distance=hit.distanceToSquared(target);
      if(distance<best){best=distance;result=hit.toArray()}
    }
    return result;
  };
}

export function moveMargin(points,index,target,radius,project,axis=null){
  const delta=target.map((v,k)=>v-points[index][k]);
  if(axis){const length=Math.hypot(...axis);if(length<1e-8)return points;const unit=axis.map(v=>v/length),amount=delta.reduce((s,v,k)=>s+v*unit[k],0);delta.forEach((_,k)=>delta[k]=unit[k]*amount)}
  const lengths=points.map((p,i)=>Math.hypot(...p.map((v,k)=>v-points[(i+1)%points.length][k])));
  const total=lengths.reduce((a,b)=>a+b,0),arc=[0];for(let i=0;i<lengths.length-1;i++)arc.push(arc.at(-1)+lengths[i]);
  return points.map((p,i)=>{const d=Math.abs(arc[i]-arc[index]),distance=Math.min(d,total-d);if(i!==index&&(radius<=0||distance>=radius))return p;
    const weight=i===index?1:(1+Math.cos(Math.PI*distance/radius))/2;
    return project(p.map((v,k)=>v+delta[k]*weight));});
}

export function smoothMargin(points,project){
  let result=points;
  for(let pass=0;pass<3;pass++)result=result.map((p,i)=>project(p.map((v,k)=>.5*v+.25*(result[(i+result.length-1)%result.length][k]+result[(i+1)%result.length][k]))));
  return result;
}

export function replaceMarginArc(points,stroke,project){
  if(stroke.length<3)throw Error('请从原边缘出发，画过需修正的区域，最后回到原边缘；至少 3 个点');
  const nearest=p=>points.reduce((best,q,i)=>{const d=Math.hypot(...p.map((v,k)=>v-q[k]));return d<best.distance?{index:i,distance:d}:best},{index:0,distance:Infinity});
  const start=nearest(stroke[0]),end=nearest(stroke.at(-1));
  if(start.distance>1||end.distance>1||start.index===end.index)throw Error('局部重画的起点和终点需靠近原边缘的两个不同位置（1 mm 内）');
  let a=start.index,b=end.index,anchors=stroke;
  const n=points.length;
  if((b-a+n)%n>n/2){[a,b]=[b,a];anchors=stroke.slice().reverse()}
  const curve=new CatmullRomCurve3([points[a],...anchors.slice(1,-1),points[b]].map(p=>new Vector3(...p)),false,'centripetal');
  const divisions=Math.max(4,Math.ceil(curve.getLength()/.3));
  const patch=curve.getPoints(divisions).slice(1,-1).map(p=>project(p.toArray()));
  const outside=[];for(let i=b;;i=(i+1)%n){outside.push(points[i]);if(i===a)break}
  const result=[...outside,...patch];
  if(result.length>512)throw Error('绘制点过多，请缩短局部重画范围');
  return result;
}
