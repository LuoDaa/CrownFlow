// Open /tests/canvasSizing.html on Vite; this fixture never reads or changes cases.
import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as THREE from 'three';
import MarginAxisPage from '../src/app/pages/MarginAxisPage';
import DesignPage from '../src/app/pages/DesignPage';
import ReviewPage from '../src/app/pages/ReviewPage';
import '../src/styles/app.css';

const geometry = new THREE.SphereGeometry(4, 24, 16);
const positions = geometry.attributes.position;
const vertices = Array.from({length:positions.count},(_,i)=>[positions.getX(i),positions.getY(i),positions.getZ(i)]);
const faces = Array.from({length:geometry.index.count/3},(_,i)=>[0,1,2].map(k=>geometry.index.getX(i*3+k)));
const normals = vertices.map(p=>new THREE.Vector3(...p).normalize().toArray());
const signed = vertices.map(p=>p[0]/20);
geometry.dispose();
const session = {vertices,faces,normals,cervical_weights:vertices.map(p=>p[2]<-2?0:1),source_session_sha256:'synthetic',
  frame:{origin:[0,0,0],x:[1,0,0],y:[0,1,0],z:[0,0,1]},state:{},configured_minimum_mm:.8};
function Fixture() {
  const [step,setStep]=useState('margin'), [view,setView]=useState({upper:true,lower:true,support:true,margin:true,scanOpacity:.25});
  const [designTool,setDesignTool]=useState('select'), [brush,setBrush]=useState({radius:1.5,strength:.1,falloff:2});
  const wf={step,setStep,view,setView,designTool,setDesignTool,brush,setBrush,caseId:'synthetic',fdi:'46',supportType:'abutment',
    busy:false,design:null,hasOuter:true,hasFinal:false,sculpt:session,margin:[],marginUndo:[],marginRedo:[],axis:[0,0,1],
    params:{occlusal_target_mm:.05,proximal_target_mm:.05,minimum_thickness_mm:.8},
    contacts:{vertices,faces,signed_mm:signed,valid:vertices.map(()=>true)},contactKind:'antagonist',
    marginConfirmed:false,axisConfirmed:false,meshVersion:0,smoothStrength:.5,
    setMsg:()=>{},loadContacts:()=>{},loadSculpt:()=>{},commitEdit:()=>Promise.resolve(),
    adaptToAntagonist:()=>{},adaptToNeighbors:()=>{},undoEdit:()=>{},redoEdit:()=>{},resetCrown:()=>{}};
  const Page={margin:MarginAxisPage,design:DesignPage,review:ReviewPage}[step];
  return <main className="cd-main">
    <label>尺寸检查（合成数据，无病例写入） <select aria-label="检查页面" value={step} onChange={e=>setStep(e.target.value)}>
      <option value="margin">Step 3</option><option value="design">Step 5</option><option value="review">Step 6</option>
    </select></label><Page wf={wf}/>
  </main>;
}
createRoot(document.getElementById('root')).render(<Fixture/>);
