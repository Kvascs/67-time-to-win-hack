import numpy as np, glob, os
files = sorted(glob.glob('C:/MosTransHack/data/npz/*.npz'))
gaps=[]; slip=[]
for f in files:
    d=np.load(f); name=os.path.basename(f)[:-4]
    fr=d['vehicle__front_bogie_velocity']; rr=d['vehicle__rear_bogie_velocity']
    gv=d['sensing__gnss__master__vel'] if d['sensing__gnss__master__vel'].shape[0]>100 else d['sensing__gnss__rover__vel']
    if len(fr)<100: continue
    for lab,a in (('F',fr),('R',rr)):
        dt=np.diff(a[:,1]); ii=np.where(dt>0.5)[0]
        for i in ii:
            t0,t1=a[i,1],a[i+1,1]
            gs=np.nan
            if len(gv)>100:
                m=(gv[:,1]>t0)&(gv[:,1]<t1)
                gs=np.hypot(gv[m,2],gv[m,3]).mean() if m.any() else np.nan
            gaps.append((name,lab,t1-t0,gs,a[i,2],a[i+1,2]))
    # slip episodes: |k v - g|>1 for >=0.5s
    if len(gv)>100:
        k=0.278 if name.startswith('30618') else 0.2763
        t=fr[:,1]; g=np.interp(t,gv[:,1],np.hypot(gv[:,2],gv[:,3])); e=k*fr[:,2]-g
        er=k*np.interp(t,rr[:,1],rr[:,2])-g
        bad=np.abs(e)>1.0
        # run lengths
        edges=np.diff(np.r_[0,bad.astype(int),0]); st=np.where(edges==1)[0]; en=np.where(edges==-1)[0]
        for s,e2 in zip(st,en):
            if t[e2-1]-t[s]>=0.3: slip.append((name, t[e2-1]-t[s], e[s:e2].max(), e[s:e2].min(), er[s:e2].max(), er[s:e2].min(), g[s:e2].mean()))
print('n gaps>0.5s', len(gaps))
for g in sorted(gaps,key=lambda x:-x[2])[:15]: print('  %s %s dur %.1fs gnss_speed %.2f v_before %.1f v_after %.1f'%g)
print('n |wheel-gnss|>1 m/s episodes >=0.3s:', len(slip))
for s in sorted(slip,key=lambda x:-x[1])[:20]: print('  %s dur %.1fs  front err max %+.2f min %+.2f rear err max %+.2f min %+.2f  v %.1f'%s)
