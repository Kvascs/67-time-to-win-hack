import sys; sys.path.insert(0,'C:/MosTransHack/research/map_matching_code')
from common import *
seen=set(); out=[]
for n in names():
    d=load(n); m=d['sensing__gnss__master__fix']; r=d['sensing__gnss__rover__fix']
    if m.shape[0]<3000: continue
    key=(m.shape[0],round(m[0,0],1))
    if key in seen: continue
    seen.add(key)
    t,p,z,st=fix_xy(m,True)
    if len(t)<0.9*m.shape[0]: continue
    tr_,pr,zr,sr=fix_xy(r,True)
    # initial heading from master->rover baseline at t0 (works at standstill)
    b=np.array([np.interp(t[0],tr_,pr[:,0]),np.interp(t[0],tr_,pr[:,1])])-p[0]; u=b/np.linalg.norm(b)
    fv=d['vehicle__front_bogie_velocity']; rv=d['vehicle__rear_bogie_velocity']
    tw=fv[:,1]; vw=0.5*(fv[:,2]+np.interp(tw,rv[:,1],rv[:,2]))/3.6
    sw=np.r_[0,np.cumsum(0.5*(vw[1:]+vw[:-1])*np.diff(tw))]
    s_at=np.interp(t,tw,sw)-np.interp(t[0],tw,sw)
    est=p[0]+s_at[:,None]*u
    e=np.linalg.norm(est-p,axis=1)
    L=s_at[-1]
    out.append((n,L,np.sqrt(np.mean(e**2)),e[-1],100*e[-1]/L,np.max(e)))
o=np.array([x[1:] for x in out])
print('runs',len(o),'mean dist %.0f m'%o[:,0].mean())
print('Straight-line heading-less DR (initial heading from dual-antenna baseline): RMSE mean %.0f m, end err mean %.0f m (%.1f%% of distance), max err mean %.0f m'%(o[:,1].mean(),o[:,2].mean(),o[:,3].mean(),o[:,4].mean()))
