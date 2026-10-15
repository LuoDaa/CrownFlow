import assert from 'node:assert/strict';
import {localBrush,localDragPreview,growMasks,transformPreview,contactColor} from '../src/sculptPreview.js';
const base=[[0,0,0],[1,0,1],[0,1,1],[2,1,1]],faces=[[0,1,2],[1,3,2]],normals=base.map(()=>[0,0,1]),locks=[0,1,1,1];
const result=localBrush(base,normals,faces,[[1,0,1]],{radius:1.2,strength:.05,operation:'add'},locks);
assert.deepEqual(result[0],base[0]);assert(result[1][2]>base[1][2]);assert.deepEqual(base[1],[1,0,1]);
const masks=growMasks(faces,[0,1,0,0],locks);assert.equal(masks[0],'FIXED');assert.equal(masks[1],'MOVING');assert.equal(masks[2],'ELASTIC');
const transformed=transformPreview(base,faces,base.map(p=>p.map(v=>v+.1)),masks,locks);assert.deepEqual(transformed[0],base[0]);assert(transformed[2][0]>0);
assert.equal(contactColor(-.15,true,.05),'#ee3d46');assert.equal(contactColor(.05,true,.05),'#30b98a');assert.equal(contactColor(.5,true,.05),'#306bc9');assert.equal(contactColor(.05,false,.05),'#9ba8ae');
for (const operation of ['add','remove','smooth','flatten']) {
  const edit = localBrush(base,normals,faces,[[1,0,1]],{radius:2,strength:.05,falloff:2,operation},locks);
  assert.deepEqual(edit[0],base[0],`${operation} keeps protected cervical vertex`);
  assert(edit.flat().every(Number.isFinite));
}

const dragged=localDragPreview(base,[[1,0,1]],[.5,0,0],{radius:1.2,falloff:2},locks);
assert.deepEqual(dragged[0],base[0],'drag keeps protected cervical vertex');
assert(dragged[1][0]>base[1][0],'drag moves the brush center');
