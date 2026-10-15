import React,{useEffect,useMemo,useRef,useState} from 'react';
import {useLoader,useThree} from '@react-three/fiber';
import {Line} from '@react-three/drei';
import {STLLoader} from 'three/examples/jsm/loaders/STLLoader.js';
import {Raycaster,Vector3} from 'three';
import {moveMargin,smoothMargin,surfaceProjector,replaceMarginArc} from '../marginEditing';

export default function MarginCurveEditor({url,points,axis,mode,radius,command,patch=[],onChange,onError}){
  const geometry=useLoader(STLLoader,url),project=useMemo(()=>surfaceProjector(geometry),[geometry]);
  const {scene,controls}=useThree(),[preview,setPreview]=useState(null),[hover,setHover]=useState(-1);
  const drag=useRef(null),handled=useRef(command?.id),shown=preview||points;
  const finish=(commit=true)=>{const current=drag.current;if(!current)return;drag.current=null;if(controls)controls.enabled=current.controlsEnabled;setPreview(null);if(commit&&current.next)onChange(current.next)};
  useEffect(()=>{setPreview(null)},[points]);
  useEffect(()=>{const cancel=e=>{if(e.key==='Escape')finish(false)};window.addEventListener('keydown',cancel);return()=>{window.removeEventListener('keydown',cancel);if(drag.current&&controls)controls.enabled=drag.current.controlsEnabled}},[controls]);
  useEffect(()=>{
    if(!command||handled.current===command.id)return;handled.current=command.id;
    if(command.kind==='patch'){try{onChange(replaceMarginArc(points,patch,project))}catch(e){onError(e.message)};return}
    if(command.kind==='smooth')onChange(smoothMargin(points,project));
    else if(command.kind==='offset'){const length=Math.hypot(...axis);if(length<1e-8)return;onChange(points.map(p=>project(p.map((v,k)=>v+axis[k]/length*command.amount))))}
  },[command]);
  const hitSupport=ray=>new Raycaster(ray.origin,ray.direction).intersectObjects(scene.children,true).find(hit=>hit.object.userData.role==='support');
  return <group>
    <Line points={[...shown,shown[0]]} color="#20a976" lineWidth={3} onClick={e=>{
      if(mode==='patch')return;
      if(drag.current||e.delta>4)return;e.stopPropagation();
      const target=project(e.point.toArray());let best=Infinity,edge=0;
      shown.forEach((p,i)=>{const a=new Vector3(...p),b=new Vector3(...shown[(i+1)%shown.length]),v=b.clone().sub(a);const t=Math.max(0,Math.min(1,new Vector3(...target).sub(a).dot(v)/(v.lengthSq()||1)));const d=a.addScaledVector(v,t).distanceToSquared(new Vector3(...target));if(d<best){best=d;edge=i}});
      if(shown.some(p=>Math.hypot(...p.map((v,k)=>v-target[k]))<.05)||shown.length>=512)return;
      onChange([...shown.slice(0,edge+1),target,...shown.slice(edge+1)]);
    }}/>
    {shown.map((p,i)=><mesh key={i} position={p} onPointerOver={e=>{e.stopPropagation();setHover(i)}} onPointerOut={()=>setHover(-1)}
      onClick={e=>{if(mode!=='patch')e.stopPropagation()}}
      onPointerDown={e=>{
        if(mode==='patch')return;
        e.stopPropagation();if(e.button!==0)return;
        if(e.nativeEvent.shiftKey){if(points.length>12)onChange(points.filter((_,j)=>j!==i));else onError('至少保留 12 个边缘点');return}
        drag.current={index:i,points:points.map(p=>p.slice()),controlsEnabled:controls?.enabled??true,next:null};
        e.target.setPointerCapture(e.pointerId);if(controls)controls.enabled=false;
      }} onPointerMove={e=>{
        const current=drag.current;if(!current||current.index!==i)return;e.stopPropagation();
        const hit=hitSupport(e.ray);if(!hit)return;
        // Limit a single gesture to the local surface, avoiding a jump onto the opposite wall.
        if(hit.point.distanceTo(new Vector3(...current.points[i]))>5)return;
        current.next=moveMargin(current.points,i,hit.point.toArray(),radius,project,mode==='updown'?axis:null);setPreview(current.next);
      }} onPointerUp={e=>{if(drag.current?.index!==i)return;e.stopPropagation();finish();e.target.releasePointerCapture(e.pointerId)}}
      onPointerCancel={()=>finish(false)} onLostPointerCapture={()=>finish(false)}>
      <sphereGeometry args={[hover===i?.20:.09,10,10]}/><meshBasicMaterial color={hover===i?'#ffffff':'#19885f'}/>
    </mesh>)}
    {patch.length>1&&<Line points={patch} color="#f2ac30" lineWidth={4}/>}
    {patch.map((p,i)=><mesh key={'patch'+i} position={p}><sphereGeometry args={[.12,10,10]}/><meshBasicMaterial color="#f2ac30"/></mesh>)}
  </group>;
}
