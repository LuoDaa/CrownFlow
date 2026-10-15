import React,{useMemo,useRef,useEffect,useState} from 'react';
import {useThree} from '@react-three/fiber';
import {TransformControls,Line} from '@react-three/drei';
import * as THREE from 'three';
import {adjacency,localBrush,localDragPreview,transformPreview,contactColor} from '../sculptPreview';
import {freeformOperation} from '../editorShortcuts';

export default function SculptCanvas({session,vertices,tool,brush,weights,masks,paintClass,gizmo,uniform,contact,contactTarget=0,thickness,failureFaces=[],onPreview,onStroke,onPaint,onTransform,busy,onMeasure,allowRemove=true,onBlocked}){
  const {controls}=useThree();
  const stroke=useRef([]),handle=useRef(),dragging=useRef(false),last=useRef(0);
  const [hover,setHover]=useState(null), operationRef=useRef(null), meshRef=useRef();
  const dragStart=useRef(null), dragDelta=useRef([0,0,0]);
  const strokeBase=useRef(null), previewRef=useRef(vertices), workingNormals=useRef(session.normals);
  useEffect(()=>{setHover(null);strokeBase.current=null;previewRef.current=vertices;workingNormals.current=session.normals;return ()=>{if(controls)controls.enabled=true}},[tool,session,controls]);
  useEffect(()=>{previewRef.current=vertices},[vertices]);
  const frame=session.frame;
  const basis=useMemo(()=>new THREE.Matrix4().makeBasis(new THREE.Vector3(...frame.x),new THREE.Vector3(...frame.y),new THREE.Vector3(...frame.z)),[session]);
  const q=useMemo(()=>new THREE.Quaternion().setFromRotationMatrix(basis),[basis]);
  const pivot=useMemo(()=>{const moving=session.vertices.filter((p,i)=>masks?masks[i]==='MOVING':weights?weights[i]>.5&&session.cervical_weights[i]>0:session.cervical_weights[i]>.999);return (weights||masks)&&moving.length?[0,1,2].map(k=>moving.reduce((s,p)=>s+p[k],0)/moving.length):frame.origin},[session,weights,masks,frame.origin]);
  const adj=useMemo(()=>adjacency(session.faces,session.vertices.length),[session.faces,session.vertices.length]);
  const geo=useMemo(()=>{
    const g=new THREE.BufferGeometry();
    g.setIndex(session.faces.flat());
    const position=new THREE.Float32BufferAttribute(session.vertices.flat(),3);position.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position',position);g.computeVertexNormals();
    return g;
  },[session]);
  useEffect(()=>()=>geo.dispose(),[geo]);
  const failureGeo=useMemo(()=>{
    const ids=(failureFaces||[]).filter(id=>Number.isInteger(id)&&id>=0&&id<session.faces.length);
    if(!ids.length)return null;
    const positions=[];
    for(const id of ids)for(const vi of session.faces[id])positions.push(...(vertices[vi]||session.vertices[vi]));
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.computeVertexNormals();return g;
  },[failureFaces,session.faces,session.vertices,vertices]);
  useEffect(()=>()=>failureGeo?.dispose(),[failureGeo]);

  const refreshNormals=()=>{
    geo.computeVertexNormals();geo.computeBoundingSphere();
    const normal=geo.attributes.normal;
    if(normal)workingNormals.current=Array.from({length:normal.count},(_,i)=>[normal.getX(i),normal.getY(i),normal.getZ(i)]);
  };

  useEffect(()=>{
    const attr=geo.attributes.position;
    let first=Infinity,lastChanged=-1;
    for(let i=0;i<vertices.length;i++){
      const p=vertices[i];
      if(Math.abs(attr.getX(i)-p[0])>1e-7||Math.abs(attr.getY(i)-p[1])>1e-7||Math.abs(attr.getZ(i)-p[2])>1e-7){
        attr.setXYZ(i,p[0],p[1],p[2]);first=Math.min(first,i);lastChanged=Math.max(lastChanged,i);
      }
    }
    if(lastChanged>=0){
      attr.clearUpdateRanges();attr.addUpdateRange(first*3,(lastChanged-first+1)*3);attr.needsUpdate=true;
      if(!dragging.current)refreshNormals();
    }
    const map=new Map(),wall=new Map();if(contact?.map)contact.sample_ids.forEach((id,i)=>map.set(id,i));if(thickness?.status==='PREVIEW_ONLY')thickness.sample_ids.forEach((id,i)=>wall.set(id,thickness.distance_mm[i]));
    const colors=vertices.flatMap((p,i)=>{let color='#f4d5a0';if(masks)color={MOVING:'#36b882',ELASTIC:'#ead050',FIXED:'#5895d9'}[masks[i]];else if(weights)color=weights[i]>.5?'#36b882':'#cbd3d7';if(contact?.signed_mm)color=contactColor(contact.signed_mm[i],contact.valid?.[i],contactTarget);if(map.has(i)){const j=map.get(i);color=contactColor(contact.map.signed_mm[j],contact.map.valid[j],contact.map.target_mm?.[j]??0)}if(wall.has(i))color=wall.get(i)<session.configured_minimum_mm?'#ee3d46':'#469bea';return new THREE.Color(color).toArray()});
    let colorAttr=geo.attributes.color;
    if(!colorAttr||colorAttr.count!==vertices.length){colorAttr=new THREE.Float32BufferAttribute(colors,3);colorAttr.setUsage(THREE.DynamicDrawUsage);geo.setAttribute('color',colorAttr)}else{colorAttr.array.set(colors);colorAttr.needsUpdate=true}
  },[vertices,masks,weights,contact,contactTarget,thickness,geo,session.configured_minimum_mm]);

  const hit=e=>new THREE.Raycaster(e.ray.origin,e.ray.direction).intersectObject(meshRef.current,false)[0];
  const emitPreview=next=>{previewRef.current=next;onPreview(next)};
  const paint=e=>{
    const h=hit(e);if(!h)return;const p=h.point.toArray();operationRef.current ||= freeformOperation(e.nativeEvent,brush.operation);
    const base=strokeBase.current||vertices||session.vertices;
    if(operationRef.current==='drag'){
      if(!dragStart.current){dragStart.current=p;stroke.current=[p];dragDelta.current=[0,0,0];}
      const raw=p.map((v,i)=>v-dragStart.current[i]);const len=Math.hypot(...raw);const delta=len>1?raw.map(v=>v/len):raw;dragDelta.current=delta;
      emitPreview(localDragPreview(base,[dragStart.current],delta,brush,session.cervical_weights,weights));return;
    }
    if(stroke.current.length>=100)return;const old=stroke.current.at(-1);if(old&&Math.hypot(...p.map((v,i)=>v-old[i]))<.2)return;stroke.current.push(p);if(paintClass){onPaint(p,paintClass);return}if(performance.now()-last.current<16)return;last.current=performance.now();if(operationRef.current==='remove'&&!allowRemove){onBlocked?.();return}
    emitPreview(localBrush(base,workingNormals.current||session.normals,session.faces,stroke.current,{...brush,operation:operationRef.current},session.cervical_weights,weights,adj));
  };

  return <>
    <mesh ref={meshRef} geometry={geo} userData={{role:'outer'}} onPointerDown={e=>{if(busy||(!tool&&!paintClass)||e.button!==0)return;const operation=freeformOperation(e.nativeEvent,brush.operation);if(operation==='remove'&&!allowRemove){onBlocked?.();return}e.stopPropagation();e.target.setPointerCapture(e.pointerId);stroke.current=[];operationRef.current=operation;dragging.current=true;strokeBase.current=(vertices||session.vertices).map(p=>[...p]);previewRef.current=strokeBase.current;if(controls)controls.enabled=false;paint(e)}} onPointerMove={e=>{const h=hit(e);if(!h){setHover(null);return}const n=h.face.normal.clone().transformDirection(h.object.matrixWorld);setHover({point:h.point.clone().addScaledVector(n,.025),q:new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,0,1),n)});onMeasure?.(h.faceIndex,h.point.toArray());if(!dragging.current)return;e.stopPropagation();paint(e)}} onPointerOut={()=>setHover(null)} onPointerUp={e=>{if(!dragging.current)return;e.stopPropagation();e.target.releasePointerCapture(e.pointerId);dragging.current=false;if(controls)controls.enabled=true;refreshNormals();if(!paintClass&&stroke.current.length){
      const base=strokeBase.current||session.vertices,finalVertices=previewRef.current||base;
      if(operationRef.current==='drag'){
        const world=new THREE.Vector3(...dragDelta.current);const local=world.applyQuaternion(q.clone().invert()).toArray();
        onStroke(stroke.current,{operation:'drag',translation:local},finalVertices,base);
      }else onStroke(stroke.current,{operation:operationRef.current},finalVertices,base);
    }stroke.current=[];operationRef.current=null;dragStart.current=null;dragDelta.current=[0,0,0];strokeBase.current=null}} onPointerCancel={()=>{dragging.current=false;stroke.current=[];operationRef.current=null;dragStart.current=null;dragDelta.current=[0,0,0];setHover(null);const base=strokeBase.current||vertices||session.vertices;emitPreview(base);strokeBase.current=null;if(controls)controls.enabled=true}}>
      <meshStandardMaterial vertexColors roughness={.45} side={THREE.DoubleSide}/>
    </mesh>
    {failureGeo&&<mesh geometry={failureGeo} raycast={()=>null} renderOrder={8}>
      <meshBasicMaterial color="#ef3340" transparent opacity={0.88} side={THREE.DoubleSide} depthTest={false} polygonOffset polygonOffsetFactor={-2}/></mesh>}
    {hover&&tool&&!busy&&<group position={hover.point} quaternion={hover.q}>
      <Line raycast={()=>null} points={Array.from({length:65},(_,i)=>[Math.cos(i*Math.PI/32)*brush.radius,Math.sin(i*Math.PI/32)*brush.radius,0])} color={tool==='remove'?'#ee3d46':tool==='smooth'?'#38c3d9':tool==='flatten'?'#a78bfa':'#2563ff'} lineWidth={2}/>
      <mesh raycast={()=>null}><sphereGeometry args={[.055,8,8]}/><meshBasicMaterial color="#ffffff"/></mesh>
    </group>}
    {gizmo&&!busy&&<TransformControls object={handle} mode={gizmo} space="local" onMouseDown={()=>{dragging.current=true;strokeBase.current=(vertices||session.vertices).map(p=>[...p]);previewRef.current=strokeBase.current}} onObjectChange={()=>{
      const h=handle.current;if(!h||!dragging.current)return;const base=strokeBase.current||vertices||session.vertices;
      const translation=h.position.clone().sub(new THREE.Vector3(...pivot)).applyQuaternion(q.clone().invert()).toArray();
      const rotation=new THREE.Euler().setFromQuaternion(q.clone().invert().multiply(h.quaternion),'XYZ');
      const scale=uniform&&gizmo==='scale'?Array(3).fill(Math.max(...h.scale.toArray())):h.scale.toArray();
      const r=new THREE.Matrix4().makeRotationFromEuler(rotation),b=basis.clone();
      const desired=base.map(p=>{const v=new THREE.Vector3(...p).sub(new THREE.Vector3(...pivot)).applyQuaternion(q.clone().invert());v.multiply(new THREE.Vector3(...scale)).applyMatrix4(r).applyMatrix4(b).add(new THREE.Vector3(...pivot)).add(new THREE.Vector3(...translation).applyMatrix4(b));return v.toArray()});
      emitPreview(transformPreview(base,session.faces,desired,masks,session.cervical_weights));
    }} onMouseUp={()=>{dragging.current=false;refreshNormals();const h=handle.current;if(!h)return;const t=h.position.clone().sub(new THREE.Vector3(...pivot)).applyQuaternion(q.clone().invert());const r=new THREE.Euler().setFromQuaternion(q.clone().invert().multiply(h.quaternion),'XYZ');const base=strokeBase.current||session.vertices;const finalVertices=previewRef.current||base;onTransform({translation:t.toArray(),rotation:[r.x,r.y,r.z].map(v=>THREE.MathUtils.radToDeg(v)),scale:uniform&&gizmo==='scale'?Array(3).fill(Math.max(...h.scale.toArray())):h.scale.toArray()},finalVertices,base);strokeBase.current=null}}><group ref={handle} position={pivot} quaternion={q}/></TransformControls>}
  </>;
}
