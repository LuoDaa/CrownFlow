import assert from 'node:assert/strict';
import {statusReducer} from '../src/workflow/status.js';
const newest={design_id:'a',design_revision:4,final_current:false};
assert.equal(statusReducer(newest,{design_id:'a',design_revision:3,final_current:true}),newest);
assert.equal(statusReducer(newest,null),null);
const other={design_id:'b',design_revision:1,final_current:false};
assert.equal(statusReducer(newest,other),other);
