import React,{useEffect,useState} from 'react';
import {useThree} from '@react-three/fiber';
import {modelBounds} from '../modelBounds';
import {useBounds} from '@react-three/drei';
import {shortcutAction} from '../viewShortcuts';
import {editorShortcut,TOOL_FROM_SHORTCUT} from '../editorShortcuts';
import {Box3,Vector3} from 'three';

export function useViewTools(onAction, enabled=true, crownEditing=false) {
  const [command,setCommand]=useState(null);
  const [wireframe,setWireframe]=useState(false),[grid,setGrid]=useState(false);
  const [help,setHelp]=useState(false);
  const act=action=>{
    if (['fit','focusSupport','top','front','right','left','bottom'].includes(action))
      setCommand(prev=>({name:action,id:(prev?.id||0)+1}));
    else if(action==='wireframe')setWireframe(v=>!v);
    else if(action==='grid')setGrid(v=>!v);
    else if(action==='help')setHelp(v=>!v);
    else if(action==='escape'){setHelp(false);onAction?.(action)}
    else return onAction?.(action)===true;
    return true;
  };
  useEffect(()=>{
    if(!enabled)return;
    const handler=e=>{if(crownEditing&&TOOL_FROM_SHORTCUT[editorShortcut(e)])return;const action=shortcutAction(e);if(action&&act(action))e.preventDefault()};
    window.addEventListener('keydown',handler);
    return()=>window.removeEventListener('keydown',handler);
  });
  return {command,wireframe,grid,help,act};
}

// Shared by both existing canvases; camera actions never modify mesh coordinates.
export function CameraCommand({command,fitKey,focusMarginKey,marginPoints=[]}) {
  const {camera,controls,invalidate,scene}=useThree();
  const bounds=useBounds();
  useEffect(()=>{if(!fitKey||focusMarginKey||!controls)return;const box=modelBounds(scene);if(!box.isEmpty())bounds.refresh(box).clip().fit()},[fitKey,focusMarginKey,controls,bounds,scene]);
  useEffect(()=>{if(!focusMarginKey||!controls||marginPoints.length<12)return;
    const box=new Box3().setFromPoints(marginPoints.map(p=>new Vector3(...p)));
    const center=box.getCenter(new Vector3()),span=box.getSize(new Vector3());
    const distance=Math.max(span.x,span.y,span.z,6)/(2*Math.tan(camera.fov*Math.PI/360))*1.35;
    const direction=camera.position.clone().sub(controls.target).normalize();
    camera.position.copy(center).addScaledVector(direction,distance);controls.target.copy(center);
    camera.lookAt(center);controls.update();invalidate();
  },[focusMarginKey,controls,camera,invalidate]);
  useEffect(()=>{
    if(!command||!controls)return;
    if(command.name==='fit'){const box=modelBounds(scene);if(!box.isEmpty())bounds.refresh(box).clip().fit();return}
    if(command.name==='focusSupport'){
      const box=modelBounds(scene,true);
      if(!box.isEmpty())bounds.refresh(box).clip().fit();return;
    }
    const direction={top:[0,0,1],front:[0,-1,0],right:[1,0,0],left:[-1,0,0],bottom:[0,0,-1]}[command.name];
    if(!direction)return;
    const distance=Math.max(camera.position.distanceTo(controls.target),1);
    camera.position.copy(controls.target).addScaledVector({x:direction[0],y:direction[1],z:direction[2]},distance);
    camera.up.set(0,command.name==='top'||command.name==='bottom'?1:0,command.name==='top'||command.name==='bottom'?0:1);
    camera.lookAt(controls.target);controls.update();invalidate();
  },[command,controls,camera,bounds,invalidate,scene]);
  return null;
}

export default function ViewTools({tools,editing=false}) {
  // 「定位牙冠」= 只把**牙冠与预备体**框进视野（不含上下颌扫描，modelBounds 的 focus 集合）。
  // 牙冠比牙列小一个量级，"适应"整场时容易小到看不见 —— 这一下专治"看不到牙冠"。
  const views=[['fit','适应','F'],['focusSupport','定位牙冠',''],['top','顶视','1'],['front','前视','2'],['right','右视','3'],['left','左视','4'],['bottom','底视','5']];
  return <>
    <div className="camera-tools" aria-label="视角与显示工具">
      {views.map(([action,label,key])=><button key={action} title={`${label}${key ? ` (${key})` : ""} · 只框住牙冠与预备体`} onClick={()=>tools.act(action)}>{label}{key && <kbd>{key}</kbd>}</button>)}
      <span className="viewsep"/>
      <button aria-pressed={tools.wireframe} className={tools.wireframe?'active':''} onClick={()=>tools.act('wireframe')} title="线框 (W)">线框<kbd>W</kbd></button>
      <button aria-pressed={tools.grid} className={tools.grid?'active':''} onClick={()=>tools.act('grid')} title="网格 (G)">网格<kbd>G</kbd></button>
      <button aria-expanded={tools.help} onClick={()=>tools.act('help')} title="快捷键 (?)">快捷键<kbd>?</kbd></button>
    </div>
    {tools.help&&<div className="shortcut-help" role="region" aria-label="快捷键帮助">
      <b>WelooAI 快捷键</b><button onClick={()=>tools.act('escape')} aria-label="关闭快捷键帮助">×</button>
      <p>左键拖动旋转 · 右键拖动平移 · 滚轮缩放视图</p>
      <p>F 适应模型 · 1–5 标准视角 · W 线框 · G 网格 · Esc 退出工具</p>
      <p>U 上颌 · L 下颌 · S 预备体 · M 边缘线显示</p>
      {editing&&<p>修正边缘时：Ctrl+Z / Backspace 撤销点 · Ctrl+Shift+Z / Ctrl+Y 恢复 · Ctrl+S 确认并保存</p>}
      <small>输入框内快捷键暂停；视角以扫描坐标系为准，视图缩放不改变 STL。</small>
    </div>}
  </>;
}
