import assert from 'node:assert/strict';
import { contactColor, contactStops, contactMetrics } from '../src/contactMap.js';
for (const target of [0,.05,.5]) {
  const stops = contactStops(target);
  for (const [value,color] of stops) assert.equal(contactColor(value,true,target),color);
  const rgb = color => [1,3,5].map(i=>parseInt(color.slice(i,i+2),16));
  for (const [value] of stops.slice(1,-1)) {
    const a=rgb(contactColor(value-1e-7,true,target)), b=rgb(contactColor(value+1e-7,true,target));
    assert(a.every((v,i)=>Math.abs(v-b[i])<=1),'no discontinuity at color stops');
  }
}
assert.equal(contactColor(null,true),'#9ba8ae');
assert.equal(contactColor(NaN,true),'#9ba8ae');
assert.equal(contactColor(-.3,false),'#9ba8ae');
const mesh={vertices:[[0,0,0],[1,0,0],[0,1,0]],faces:[[0,1,2]],signed_mm:[.05,-100,.05],valid:[true,false,true]};
const result=contactMetrics(mesh,.05);
assert.equal(result.penetration,0,'invalid negative value must not create penetration');
assert.equal(result.minimum,.05);
assert.equal(result.area,1/3,'only valid in-band vertex area contributes');
assert.equal(contactMetrics({...mesh,valid:[false,false,false]},.05),null);
assert.equal(contactMetrics({...mesh,signed_mm:null},.05),null,'unsigned distance is not a contact result');
assert.equal(contactMetrics(mesh,null),null,'unconfirmed target cannot define contact area');
