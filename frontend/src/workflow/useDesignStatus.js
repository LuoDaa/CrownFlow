import {useEffect,useReducer} from 'react';
import {api} from '../api';
import {statusReducer} from './status';

export default function useDesignStatus(design,busy,onError){
  const [state,update]=useReducer(statusReducer,null);
  useEffect(()=>{
    if(!design?.id){update(null);return;}
    let active=true;
    api.designStatus(design.id).then(value=>{if(active)update(value)}).catch(e=>{if(active)onError(e.message)});
    return()=>{active=false};
  },[design?.id,design?.design_revision,busy]);
  return [state?.design_id===design?.id&&state?.design_revision===design?.design_revision?state:null,update];
}
