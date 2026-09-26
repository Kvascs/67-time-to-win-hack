import numpy as np, glob, os
files = sorted(glob.glob('C:/MosTransHack/data/npz/*.npz'))
L=[]; ang=[]; st=[]
for f in files:
    d=np.load(f); m=d['sensing__gnss__master__fix']; r=d['sensing__gnss__rover__fix']; gv=d['sensing__gnss__master__vel']
    if len(m)<100 or len(r)<100 or len(gv)<100: continue
    t=m[:,1]; lat0=m[0,2]
    rx=np.interp(t,r[:,1],r[:,3]); ry=np.interp(t,r[:,1],r[:,2])
    dx=(rx-m[:,3])*111320*np.cos(np.radians(lat0)); dy=(ry-m[:,2])*110540
    vx=np.interp(t,gv[:,1],gv[:,2]); vy=np.interp(t,gv[:,1],gv[:,3]); sp=np.hypot(vx,vy)
    k=sp>3
    if k.sum()<50: continue
    L.append(np.median(np.hypot(dx,dy)))
    base=np.arctan2(dy[k],dx[k]); head=np.arctan2(vy[k],vx[k])
    dd=np.angle(np.exp(1j*(base-head)))
    ang.append(np.degrees(np.median(np.abs(dd))))
    st.append(m[:,5].mean())
    # stationary spread
print('master-rover baseline m: median', np.median(L), 'pct10/90', np.percentile(L,[10,90]))
print('angle |baseline - velocity heading| deg median per bag: pct10/50/90', np.percentile(ang,[10,50,90]))
print('fix status mean (2=GBAS/RTK?)', np.percentile(st,[10,50,90]))
