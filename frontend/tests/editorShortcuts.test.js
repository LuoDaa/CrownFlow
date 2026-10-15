import assert from 'node:assert/strict';
import {editorShortcut,freeformOperation,TOOL_FROM_SHORTCUT} from '../src/editorShortcuts.js';
assert.equal(editorShortcut({key:'A'}),'add');
assert.equal(editorShortcut({key:'2'}),'rotate');
assert.equal(editorShortcut({key:'S',ctrlKey:true}),null);
assert.equal(editorShortcut({key:'1',target:{closest:()=>true}}),null);
assert.equal(editorShortcut({key:'D',repeat:true}),null);
// 修饰键只在输入动作层解释一次
assert.equal(freeformOperation({},'add'),'add');
assert.equal(freeformOperation({shiftKey:true},'add'),'remove');
assert.equal(freeformOperation({ctrlKey:true,shiftKey:true},'add'),'smooth');
assert.equal(freeformOperation({metaKey:true,shiftKey:true},'remove'),'smooth');
assert.equal(freeformOperation({shiftKey:true},'smooth'),'add');
assert.equal(freeformOperation({shiftKey:true},'flatten'),'smooth');
// 键盘动作映射到 Design 页工具枚举
assert.equal(TOOL_FROM_SHORTCUT[editorShortcut({key:'a'})],'add');
assert.equal(TOOL_FROM_SHORTCUT[editorShortcut({key:'1'})],'move');
assert.equal(TOOL_FROM_SHORTCUT['drag'],'drag','D 应映射到局部拖动工具');
