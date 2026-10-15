import assert from 'node:assert/strict';
import {PlaneGeometry} from 'three';
import {moveMargin,smoothMargin,surfaceProjector,replaceMarginArc} from '../src/marginEditing.js';

const project=surfaceProjector(new PlaneGeometry(20,20));
assert.deepEqual(project([1,2,3]),[1,2,0]);
const ring=Array.from({length:64},(_,i)=>[4*Math.cos(i*Math.PI/32),4*Math.sin(i*Math.PI/32),0]);
const moved=moveMargin(ring,0,[4.5,0,0],1.5,project);
assert.ok(moved[1][0]>ring[1][0]); // Hidden/dense neighbors must move with the handle.
assert.ok(moved[63][0]>ring[63][0]); // Same behavior across the loop seam.
assert.deepEqual(moved[32],ring[32]);
assert.deepEqual(ring[0],[4,0,0]); // Drag previews must not mutate the undo snapshot.
const isolated=moveMargin(ring,0,[4.5,0,0],0,project);
assert.deepEqual(isolated[1],ring[1]);
const axial=moveMargin(ring,0,[4.5,0,2],1.5,p=>p,[0,0,1]);
assert.equal(axial[0][0],4);assert.equal(axial[0][2],2);
const jagged=ring.map((p,i)=>[p[0],p[1],i%2?.4:-.4]);
const smoothed=smoothMargin(jagged,project);
assert.equal(smoothed.length,64);assert.ok(smoothed.every(p=>p[2]===0));
assert.equal(jagged[0][2],-.4);
const patch=replaceMarginArc(ring,[ring[60],[4.5,0,0],ring[4]],project);
assert.ok(patch.some(p=>p[0]>4.2));
assert.ok(patch.includes(ring[32]));
assert.ok(!patch.includes(ring[0]));
assert.throws(()=>replaceMarginArc(ring,[[20,20,0],[21,21,0],ring[4]],project),/1 mm/);
