export function statusReducer(current,incoming){
  if(!incoming)return null;
  if(current?.design_id===incoming.design_id&&current.design_revision>incoming.design_revision)return current;
  return incoming;
}
