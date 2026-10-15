import React,{Suspense,useEffect,useRef,useState} from "react";
import {Canvas,useLoader,useThree} from "@react-three/fiber";
import {Bounds,OrbitControls,Line,TransformControls} from "@react-three/drei";
import {STLLoader} from "three/examples/jsm/loaders/STLLoader.js";
import * as THREE from "three";
import {CameraCommand} from './ViewTools';
import MarginCurveEditor from './MarginCurveEditor';

export class MeshViewportBoundary extends React.Component{
  state={error:null};
  static getDerivedStateFromError(error){return {error}}
  componentDidCatch(error){this.props.onFailure?.(error)}
  render(){
    if(!this.state.error)return this.props.children;
    // ⚠️ three 的 loader 会把**失败的 promise** 也缓存住，所以"重新载入"必须换 URL
    // （页面级 meshVersion 会给所有网格 URL 加新的 ?v=），不能只重挂组件。
    return <div role="alert" className="msg">
      <b>3D 视图加载失败</b>
      <div>{this.state.error.message}</div>
      <div className="msg-actions">
        {this.props.onRetry&&<button type="button" className="cd-btn secondary"
          onClick={()=>{this.setState({error:null});this.props.onRetry();}}>重新载入模型</button>}
      </div>
    </div>;
  }
}

/** 所有异步几何（STL / 编辑器）就绪后挂载一次 —— 视口的"加载中"遮罩靠它收起。 */
function ReadySignal({onChange}){
  React.useEffect(()=>{onChange(true);return ()=>onChange(false)},[onChange]);
  return null;
}

function Stl({url,color,opacity=1,role,source,onPick,pickRole,wireframe=false,axis,margin,undercut=false,selection,pickRadius=1,matrix,onProbe,probeMinMs=150}){
  const geo=useLoader(STLLoader,url);
  const probeAt=useRef(0);
  useEffect(()=>geo.computeVertexNormals(),[geo]);
  const shown=React.useMemo(()=>{if(selection && selection.source===source&&selection.face_ids?.length){const g=geo.clone(),selected=new Set(selection.face_ids);const colors=[];for(let i=0;i<g.attributes.position.count;i++){const c=new THREE.Color(selected.has(Math.floor(i/3))?'#ff5a45':'#ffffff');colors.push(c.r,c.g,c.b)}g.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));return g;}if(!undercut||!axis||!margin?.length)return geo;
    const g=geo.clone();g.computeVertexNormals();const n=g.attributes.normal,p=g.attributes.position;
    const a=new THREE.Vector3(...axis).normalize(),c=new THREE.Vector3(...margin.reduce((s,q)=>s.map((v,i)=>v+q[i]/margin.length),[0,0,0]));
    const colors=[];for(let i=0;i<p.count;i++){const dot=new THREE.Vector3().fromBufferAttribute(n,i).dot(a),height=new THREE.Vector3().fromBufferAttribute(p,i).sub(c).dot(a);
      const col=new THREE.Color(height>0&&dot<-.02?'#ef6b62':'#70c3b0');colors.push(col.r,col.g,col.b)}
    g.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));return g;
  },[geo,undercut,axis?.join(','),margin,selection,source]);
  useEffect(()=>()=>{if(shown!==geo)shown.dispose()},[shown,geo]);
  return <mesh geometry={shown} matrix={matrix?new THREE.Matrix4().set(...matrix.flat()):undefined} matrixAutoUpdate={!matrix} userData={{role}} onClick={e=>{
    if(role!==pickRole||e.delta>4)return;
    e.stopPropagation();
    const ids=[];if(selection && selection.source===source){const p=geo.attributes.position,local=e.object.worldToLocal(e.point.clone());for(let i=0;i<p.count;i+=3){const c=new THREE.Vector3().fromBufferAttribute(p,i).add(new THREE.Vector3().fromBufferAttribute(p,i+1)).add(new THREE.Vector3().fromBufferAttribute(p,i+2)).divideScalar(3);if(c.distanceTo(local)<=pickRadius)ids.push(i/3)}}
    const normal=e.face?.normal?.clone().transformDirection(e.object.matrixWorld).toArray();
    onPick?.(role,[e.point.x,e.point.y,e.point.z],source,{face_ids:ids,normal,face_id:e.faceIndex});
  }} onPointerMove={e=>{
    if(!onProbe)return;
    const now=performance.now();if(now-probeAt.current<probeMinMs)return;probeAt.current=now;
    const normal=e.face?.normal?.clone().transformDirection(e.object.matrixWorld).toArray();
    onProbe([e.point.x,e.point.y,e.point.z],normal);
  }}>
    <meshStandardMaterial vertexColors={shown!==geo} wireframe={wireframe} color={shown!==geo?'#ffffff':color} roughness={role==='final'?0.42:0.6} side={THREE.DoubleSide}
      transparent={opacity<1} opacity={opacity}/>
  </mesh>
}

function KeyLight({headlight,fallback=[50,-40,90]}){
  // 「从视角打光」：按下的那一刻把光源方向**定格**在当前视角。
  // 之后转到侧面看台肩时，阴影会把颈缘显出来（exocad Adjust Light from view 的做法）。
  const camera=useThree(s=>s.camera);
  const [pos,setPos]=useState(null);
  useEffect(()=>{
    if(!headlight){setPos(null);return}
    const p=camera.position.clone();
    if(p.length()<1e-6)p.set(0,0,1);
    setPos(p.normalize().multiplyScalar(120).toArray());
  },[headlight]);
  return <directionalLight position={pos||fallback} intensity={2.0}/>;
}

function AxisGuide({axis,margin,onChange}){
  const handle=useRef(),center=new THREE.Vector3(...margin.reduce((s,p)=>s.map((v,i)=>v+p[i]/margin.length),[0,0,0]));
  const end=center.clone().addScaledVector(new THREE.Vector3(...axis).normalize(),8);
  return <><Line points={[center.toArray(),end.toArray()]} color="#55b6ff" lineWidth={3}/><TransformControls object={handle} mode="translate" onMouseUp={()=>{const a=handle.current.position.clone().sub(center);if(a.length()>.1)onChange(a.normalize().toArray())}}><mesh ref={handle} position={end} userData={{role:'axis-handle'}}><sphereGeometry args={[.6,16,16]}/><meshStandardMaterial color="#55b6ff"/></mesh></TransformControls></>;
}
function Dot({p,color,onClick,onDrag,radius=.25}){
  const {scene,controls}=useThree(),ref=useRef(),drag=useRef(null);
  return <mesh ref={ref} position={p} onClick={e=>{if(e.delta<=4)onClick?.(e)}} onPointerDown={e=>{
    if(!onDrag||e.button!==0||e.nativeEvent.shiftKey)return;e.stopPropagation();drag.current=null;
    e.target.setPointerCapture(e.pointerId);if(controls)controls.enabled=false;
  }} onPointerMove={e=>{
    if(!onDrag||!e.buttons||e.nativeEvent.shiftKey)return;e.stopPropagation();
    const ray=new THREE.Raycaster(e.ray.origin,e.ray.direction);
    const hit=ray.intersectObjects(scene.children,true).find(h=>h.object.userData.role==='support');
    if(hit){ref.current.position.copy(hit.point);drag.current=hit.point.toArray();}
  }} onPointerUp={e=>{
    if(!onDrag)return;e.stopPropagation();e.target.releasePointerCapture(e.pointerId);if(controls)controls.enabled=true;
    if(drag.current){onDrag(drag.current);drag.current=null;}
  }}><sphereGeometry args={[radius,12,12]}/><meshStandardMaterial color={color}/></mesh>;
}

export default function Viewer({mode,upper,lower,support,template,outer,finalMesh,cutter,neighborMeshes=[],
  onMarginCurveChange,marginMode,marginPatch,marginRadius,marginCommand,onMarginError,pickRole,onPick,onMarginSelect,onMarginDelete,onMarginDrag,insertionAxis,onAxisChange,showAxis=false,marginPoints=[],marginDrafting=false,focusMarginKey=null,supportCenter,implantMarker,tools,scanOpacity=.62,showMargin=true,selection,pickRadius,antagonistTransform,workingJaw,sceneKey,editorScene,marginConfirmed=false,onSectionProbe,headlight=false,onRetryMesh}){
  const showPatient=mode!=="template"&&!editorScene?.hideScans;
  // 视口"加载中"状态：只要这一批网格还有没就绪的（Suspense 仍挂着）就显示遮罩，
  // 避免别的电脑上出现"空白画布 = 生成失败"的误判。
  const meshKey=[upper,lower,support,template,outer,finalMesh,cutter,
    ...neighborMeshes.map(mesh=>mesh.url),editorScene?"editor":""].join('|');
  const hasMeshes=!!(upper||lower||support||template||outer||finalMesh||cutter||neighborMeshes.length);
  const [loadedKey,setLoadedKey]=useState(null);
  const [failedKey,setFailedKey]=useState(null);
  const loading=hasMeshes&&loadedKey!==meshKey&&failedKey!==meshKey;
  return <div className="viewer">
    <MeshViewportBoundary key={sceneKey} onFailure={()=>setFailedKey(meshKey)}
      onRetry={()=>{setFailedKey(null);setLoadedKey(null);onRetryMesh?.()}}><Canvas dpr={[1,2]} resize={{debounce:0}} camera={{position:[0,-115,65],fov:42}}>
      <color attach="background" args={["#eef1f3"]}/>
      <ambientLight intensity={1.15}/><KeyLight headlight={headlight}/>
      <Suspense fallback={null}>
        <ReadySignal key={meshKey} onChange={(ready)=>{if(ready)setLoadedKey(meshKey)}}/>
        <Bounds margin={1.3}>
          {showPatient&&upper&&<Stl url={upper} source="upper" selection={selection} pickRadius={pickRadius} matrix={workingJaw==='lower'?antagonistTransform:undefined} wireframe={tools?.wireframe} role="patient" pickRole={pickRole} onPick={onPick} color="#d8c5b2" opacity={scanOpacity}/>}
          {showPatient&&lower&&<Stl url={lower} source="lower" selection={selection} pickRadius={pickRadius} matrix={workingJaw==='upper'?antagonistTransform:undefined} wireframe={tools?.wireframe} role="patient" pickRole={pickRole} onPick={onPick} color="#c1a58e" opacity={scanOpacity}/>}
          {showPatient&&support&&<Stl url={support} wireframe={tools?.wireframe} role="support" pickRole={pickRole} onPick={onPick} color="#6fc0b9" opacity={onMarginCurveChange?1:.45} axis={insertionAxis} margin={marginPoints} undercut={showAxis} onProbe={onSectionProbe}/>}
          {showPatient&&neighborMeshes.map(({side,url})=><Stl key={url} url={url} role="neighbor" wireframe color={side==='mesial'?'#62dcf5':'#f5a0e1'}/>)}
          {mode==="template"&&template&&<Stl url={template} wireframe={tools?.wireframe} role="template" pickRole={pickRole} onPick={onPick} color="#ef927f"/>}
          {editorScene?.content}
          {mode==="outer"&&outer&&!editorScene&&<Stl url={outer} wireframe={tools?.wireframe} role="outer" color="#f0be83"/>}
          {mode==="final"&&finalMesh&&!editorScene&&<Stl url={finalMesh} wireframe={tools?.wireframe} role="final" color="#eee9df"/>}
          {mode==="cutter"&&cutter&&!editorScene&&<Stl url={cutter} wireframe={tools?.wireframe} role="cutter" color="#799bd8" opacity={.72}/>}
          {!onMarginCurveChange&&mode!=="template"&&showMargin&&marginPoints.map((p,i)=><Dot key={"m"+i} p={p} radius={marginDrafting?.22:.12} color={marginConfirmed?'#24ad77':'#dba83a'} onDrag={onMarginDrag?q=>onMarginDrag(i,q):undefined} onClick={e=>{e.stopPropagation();if(e.nativeEvent.shiftKey)onMarginDelete?.(i);else onMarginSelect?.(i)}}/>)}
          {!onMarginCurveChange&&mode!=="template"&&showMargin&&marginPoints.length>1&&<Line points={marginDrafting||marginPoints.length<12?marginPoints:[...marginPoints,marginPoints[0]]} color={marginConfirmed?"#24ad77":"#dba83a"} lineWidth={3}/>}
          {onMarginCurveChange&&support&&showMargin&&<MarginCurveEditor url={support} points={marginPoints} axis={insertionAxis} mode={marginMode} radius={marginRadius} command={marginCommand} patch={marginPatch} onChange={onMarginCurveChange} onError={onMarginError}/>}
          {mode!=="template"&&supportCenter&&<Dot p={supportCenter} color="#f06f89"/>}
          {showPatient&&implantMarker?.top_center&&<><Line points={[implantMarker.base_center,implantMarker.top_center]} color="#168d85" lineWidth={4}/><Dot p={implantMarker.top_center} radius={.55} color="#168d85"/></>}
          <CameraCommand command={tools?.command} fitKey={[sceneKey,mode,upper,lower,support,template,outer,finalMesh,cutter].join('|')} focusMarginKey={focusMarginKey} marginPoints={marginPoints}/>
          {mode!=="template"&&showAxis&&marginPoints.length>2&&<AxisGuide axis={insertionAxis} margin={marginPoints} onChange={onAxisChange}/>}
        </Bounds>
      </Suspense>
      <OrbitControls makeDefault enableRotate={!editorScene?.painting||editorScene?.captureOnlyOnMesh} enablePan={!editorScene?.painting||editorScene?.captureOnlyOnMesh}/>
      {tools?.grid&&<gridHelper args={[160,32,"#4f5968","#333c47"]} rotation={[Math.PI/2,0,0]}/>}
    </Canvas></MeshViewportBoundary>
    {loading&&<div className="cd-viewport-loading" role="status" aria-live="polite">
      <span className="cd-spinner" aria-hidden="true"/>
      <b>正在加载 3D 模型…</b>
      <small>{mode==="final"?"正在载入最终牙冠":mode==="outer"?"正在载入牙冠外形":"正在载入牙列与牙冠"}</small>
    </div>}
  </div>;
}
