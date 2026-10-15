/** 塑形输入动作层：键盘选工具 + 修饰键 → 实际操作，统一在这里解释。 */

export function editorShortcut(e){
  if(e.ctrlKey||e.metaKey||e.altKey||e.shiftKey||e.repeat||e.isComposing||e.target?.closest?.('input,select,textarea,[contenteditable]'))return null;
  return {'1':'translate','2':'rotate','3':'scale',a:'add',r:'remove',s:'smooth',d:'drag'}[e.key.toLowerCase()]||null;
}

/** 键盘动作 → Design 页的工具名（页面只认自己的工具枚举）。 */
export const TOOL_FROM_SHORTCUT={translate:'move',rotate:'rotate',add:'add',remove:'remove',smooth:'smooth',drag:'drag'};

/**
 * Free-forming 修饰键 → 本次笔刷真正执行的操作。
 * 放在这里而不是写进各组件，是为了让 Shift / Ctrl+Shift 的含义只有一处定义（§13）。
 */
export function freeformOperation(event,operation){
  const {shiftKey,ctrlKey,metaKey}=event||{};
  if((ctrlKey||metaKey)&&shiftKey)return 'smooth';
  if(shiftKey&&operation==='add')return 'remove';
  if(shiftKey&&operation==='smooth')return 'add';
  if(shiftKey&&operation==='flatten')return 'smooth';
  return operation;
}
