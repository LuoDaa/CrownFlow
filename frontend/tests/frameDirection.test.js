import assert from 'node:assert/strict';
import { suggestPatientFrame } from '../src/domain/frameDirection.js';

const normalize = v => { const n = Math.hypot(...v); return v.map(x => x / n); };
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
const axis = normalize([0.036977,0.991563,-0.124236]);
const ref = [[1,0,0],[0,1,0],[0,0,1]].sort((a,b)=>Math.abs(dot(a,axis))-Math.abs(dot(b,axis)))[0];
const u = normalize(cross(axis,ref)); const v = cross(axis,u);
const ring = Array.from({length:128},(_,i)=>{
  const t=2*Math.PI*i/128;
  return u.map((x,j)=>x*4.25*Math.cos(t)+v[j]*4.05*Math.sin(t)+[7,-3,12][j]);
});
const frame = suggestPatientFrame(ring,axis);
assert.ok(Math.abs(dot(frame.mesial,axis))<1e-10);
assert.ok(Math.abs(dot(frame.mesial,u))>.999999);
assert.ok(dot(frame.buccal,cross(axis,frame.mesial))>.999999);
assert.throws(()=>suggestPatientFrame(Array.from({length:64},(_,i)=>{
  const t=2*Math.PI*i/64;
  return u.map((x,j)=>x*4*Math.cos(t)+v[j]*4*Math.sin(t));
}),axis),/跨度相近/);
