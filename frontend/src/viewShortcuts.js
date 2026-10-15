export function shortcutAction(event) {
  if (event.defaultPrevented || event.repeat || event.isComposing ||
      event.target?.closest?.('input,textarea,select,[contenteditable]:not([contenteditable="false"]),[role="textbox"]')) return null;
  const key = event.key.toLowerCase();
  if (event.ctrlKey || event.metaKey) {
    if (event.altKey) return null;
    if (key === 's') return 'save';
    if (key === 'z') return event.shiftKey ? 'redo' : 'undo';
    if (key === 'y') return 'redo';
    return null;
  }
  if (event.altKey) return null;
  return ({f:'fit', '1':'top', '2':'front', '3':'right', '4':'left', '5':'bottom',
    w:'wireframe', g:'grid', '?':'help', escape:'escape', u:'upper', l:'lower',
    s:'support', m:'margin', i:'intaglio', o:'outer', t:'thin', c:'contact',
    a:'occlusal', backspace:'undo'})[key] || null;
}
