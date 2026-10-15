import {Box3} from 'three';
const all=new Set(['patient','support','outer','final','cutter','template','neighbor','axis-handle']);
const focused=new Set(['support','outer','final','cutter','axis-handle']);
export function modelBounds(scene,focus=false){
  scene.updateMatrixWorld(true);
  const box=new Box3(),roles=focus?focused:all;
  scene.traverse(object=>{if(object.visible&&object.geometry&&roles.has(object.userData.role))box.expandByObject(object)});
  return box;
}
