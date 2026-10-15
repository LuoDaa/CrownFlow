export const MIN_MARGIN_POINTS=12;
export function canConfirmMargin(points,reviewed){
  if(!reviewed||points.length<MIN_MARGIN_POINTS)return false;
  if(!points.every(p=>p.length===3&&p.every(Number.isFinite)))return false;
  const center=points.reduce((s,p)=>s.map((v,i)=>v+p[i]/points.length),[0,0,0]);
  const area=points.reduce((s,p,i)=>{const q=points[(i+1)%points.length];return s.map((v,k)=>v+(p[(k+1)%3]-center[(k+1)%3])*(q[(k+2)%3]-center[(k+2)%3])-(p[(k+2)%3]-center[(k+2)%3])*(q[(k+1)%3]-center[(k+1)%3]))},[0,0,0]);
  return Math.hypot(...area)>0.01
    &&new Set(points.map(p=>JSON.stringify(p))).size>=MIN_MARGIN_POINTS;
}
