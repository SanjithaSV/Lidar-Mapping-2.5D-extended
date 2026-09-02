from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

@dataclass
class EgoMotion:
    tx: float=0.; ty: float=0.; yaw: float=0.; speed_mps: float=0.; speed_kmh: float=0.; confidence: float=0.; rmse: float=0.; inlier_ratio: float=0.; dt: float=.1
    def as_dict(self): return asdict(self)

def _xy(pts): return np.asarray(pts)[:,:2].astype(np.float64,copy=False)
def _transform_xy(xy,tx,ty,yaw):
    c,s=np.cos(yaw),np.sin(yaw); return xy@np.array([[c,-s],[s,c]]).T+np.array([tx,ty])
def _phase_translation(prev_xy,cur_xy,cell=.20,extent=60.):
    n=max(int(np.ceil(2*extent/cell)),32)
    def raster(x):
        ix=np.floor((x[:,0]+extent)/cell).astype(int); iy=np.floor((x[:,1]+extent)/cell).astype(int); m=(ix>=0)&(ix<n)&(iy>=0)&(iy<n); z=np.zeros((n,n),np.float32); np.add.at(z,(iy[m],ix[m]),1); return z
    A,B=raster(prev_xy),raster(cur_xy); R=np.fft.fft2(A)*np.conj(np.fft.fft2(B)); R/=np.maximum(np.abs(R),1e-8); cc=np.abs(np.fft.ifft2(R)); iy,ix=np.unravel_index(np.argmax(cc),cc.shape); ix=ix-n if ix>n//2 else ix; iy=iy-n if iy>n//2 else iy; return np.array([ix*cell,iy*cell])
def estimate_ego_motion(prev_pts,cur_pts,dt=.1,max_iter=12,max_corr=1.5):
    P,Q=_xy(prev_pts),_xy(cur_pts)
    if len(P)<30 or len(Q)<30:return None
    init=_phase_translation(P,Q); tx,ty=float(init[0]),float(init[1]); yaw=0.; tree=cKDTree(P)
    for _ in range(max_iter):
        T=_transform_xy(Q,tx,ty,yaw); d,idx=tree.query(T); m=d<max_corr
        if m.sum()<20:break
        X,Y=Q[m],P[idx[m]]; H=(X-X.mean(0)).T@(Y-Y.mean(0)); U,_,Vt=np.linalg.svd(H); R=Vt.T@U.T
        if np.linalg.det(R)<0:Vt[-1]*=-1;R=Vt.T@U.T
        da=np.arctan2(R[1,0],R[0,0]); sh=Y.mean(0)-R@X.mean(0); tx=.5*tx+.5*sh[0];ty=.5*ty+.5*sh[1];yaw=.5*yaw+.5*da
    T=_transform_xy(Q,tx,ty,yaw);d,_=tree.query(T);m=d<max_corr;rmse=float(np.sqrt(np.mean(d[m]**2))) if m.any() else float('inf');inlier=float(m.mean());conf=float(np.clip(.55*inlier+.45*np.exp(-rmse/.5),0,1));sp=float(np.hypot(tx,ty))/max(float(dt),1e-6);return EgoMotion(float(tx),float(ty),float(yaw),sp,sp*3.6,conf,rmse,inlier,float(dt))
def object_motion(prev_pts,cur_pts,ego,cur_cluster_ids,cur_cluster_classes,residual_base=.35,residual_frac=.015):
    if ego is None or cur_cluster_ids is None or cur_cluster_classes is None:return []
    P,C=_xy(prev_pts),_xy(cur_pts);tree=cKDTree(_transform_xy(P,ego.tx,ego.ty,ego.yaw));d,_=tree.query(C);ids=np.asarray(cur_cluster_ids);cls=np.asarray(cur_cluster_classes);out=[]
    for cid in np.unique(ids):
        if cid<0:continue
        m=ids==cid; vals=d[m];xyz=np.asarray(cur_pts)[m,:3];center=xyz.mean(0);thr=max(residual_base,float(np.hypot(center[0],center[1]))*residual_frac);med=float(np.median(vals));p75=float(np.percentile(vals,75));frac=float(np.mean(vals>thr));state='UNKNOWN' if len(vals)<8 else ('DYNAMIC' if med>thr or frac>.35 else 'STATIC');c=int(np.bincount(cls[m].astype(int)).argmax())
        out.append({'cluster_id':int(cid),'class_id':c,'state':state,'points':int(m.sum()),'median_m':med,'p75_m':p75,'moving_fraction':frac,'center':center.tolist(),'size':np.ptp(xyz,axis=0).tolist()})
    return out
def _trajectory_stats(traj):
    a=np.asarray(traj,float)
    if len(a)<2:return {'history_len':len(a),'displacement_m':0.,'velocity_xy_mps':[0.,0.],'speed_mps':0.,'speed_kmh':0.,'velocity_std_mps':0.,'direction_consistency':0.,'trajectory_rmse_m':0.}
    v=np.diff(a,axis=0);vel=v.mean(0);norms=np.linalg.norm(v,axis=1);dirs=v/(norms[:,None]+1e-9);t=np.arange(len(a),dtype=float);A=np.c_[t,np.ones(len(t))];pred=np.c_[A@np.linalg.lstsq(A,a[:,0],rcond=None)[0],A@np.linalg.lstsq(A,a[:,1],rcond=None)[0]];return {'history_len':len(a),'displacement_m':float(np.linalg.norm(a[-1]-a[0])),'velocity_xy_mps':vel.tolist(),'speed_mps':float(np.linalg.norm(vel)),'speed_kmh':float(np.linalg.norm(vel))*3.6,'velocity_std_mps':float(np.std(norms)),'direction_consistency':float(np.clip(np.linalg.norm(dirs.mean(0)),0,1)),'trajectory_rmse_m':float(np.sqrt(np.mean(np.sum((a-pred)**2,axis=1))))}
def track_objects(prev_objects,cur_objects,ego,next_track_id=1,gate=4.,dt=.1,history_window=6,dynamic_speed=.75):
    if not cur_objects:return [],next_track_id
    prev=list(prev_objects or []);cur=list(cur_objects);pred=[]
    for p in prev:
        c=np.asarray(p.get('center',[0,0,0]),float);pred.append(_transform_xy(c[None,:2],ego.tx,ego.ty,ego.yaw)[0] if ego else c[:2])
    cost=np.full((len(prev),len(cur)),1e6)
    for i,p in enumerate(prev):
        for j,q in enumerate(cur):
            if int(p.get('class_id',-1))==int(q.get('class_id',-2)):cost[i,j]=np.linalg.norm(pred[i]-np.asarray(q.get('center',[0,0,0]))[:2])
    pairs=[]
    if prev:
        rows,cols=linear_sum_assignment(cost);pairs=[(i,j,float(cost[i,j])) for i,j in zip(rows,cols) if cost[i,j]<=gate]
    used=set();out=[]
    for i,j,dist in pairs:
        used.add(j);q=dict(cur[j]);p=prev[i];c=np.asarray(q['center'],float);pc=np.asarray(p['center'],float);prevxy=_transform_xy(pc[None,:2],ego.tx,ego.ty,ego.yaw)[0] if ego else pc[:2];hist=[np.asarray(x,float) for x in p.get('trajectory_xy',[])][-history_window:]+[c[:2]];ts=_trajectory_stats(hist);rv=(c[:2]-prevxy)/max(dt,1e-6);q.update(track_id=int(p.get('track_id',0)),age=int(p.get('age',1))+1,displacement_xy_m=(c[:2]-prevxy).tolist(),relative_velocity_mps=rv.tolist(),relative_speed_mps=float(np.linalg.norm(rv)),association_distance_m=dist,trajectory_xy=np.asarray(hist).tolist(),trajectory=ts)
        if q['age']>=2 and ts['speed_mps']>dynamic_speed and ts['direction_consistency']>.65:q['state']='DYNAMIC'
        out.append(q)
    for j,q0 in enumerate(cur):
        if j in used:continue
        q=dict(q0);c=np.asarray(q.get('center',[0,0,0]),float);q.update(track_id=next_track_id,age=1,displacement_xy_m=[0,0],relative_velocity_mps=[0,0],relative_speed_mps=0.,association_distance_m=None,trajectory_xy=[c[:2].tolist()],trajectory=_trajectory_stats([c[:2]]));next_track_id+=1;out.append(q)
    return out,next_track_id
