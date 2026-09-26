import numpy as np, glob, os
files = sorted(glob.glob('C:/MosTransHack/data/npz/*.npz'))
res = {}
lat_all=[]; ooo=0; tot=0
rows=[]
for f in files:
    d = np.load(f)
    name=os.path.basename(f)[:-4]; veh=name.split('_')[0]
    fr=d['vehicle__front_bogie_velocity']; rr=d['vehicle__rear_bogie_velocity']; cmd=d['vehicle__driver_position_cmd']
    gv=d['sensing__gnss__master__vel'] if d['sensing__gnss__master__vel'].shape[0]>100 else d['sensing__gnss__rover__vel']
    if fr.shape[0]<100 or gv.shape[0]<100 or rr.shape[0]<100: continue
    for a in (fr,rr,cmd):
        lat_all.append(a[:,0]-a[:,1]); ooo+=np.sum(np.diff(a[:,1])<0); tot+=len(a)
    # GNSS speed at wheel stamps
    gs=np.hypot(gv[:,2],gv[:,3]); 
    t=fr[:,1]
    g=np.interp(t,gv[:,1],gs)
    vr=np.interp(t,rr[:,1],rr[:,2]); vf=fr[:,2]
    m=(g>3)&(vf>3)&(vr>3)
    if m.sum()<100: continue
    # ratio (m/s per unit)
    kf=np.median(g[m]/vf[m]); kr=np.median(g[m]/vr[m])
    diff=(vf-vr)
    rows.append((veh,name,kf,kr,np.percentile(np.abs(diff[m]),[50,95,99,99.9]), len(t)))
lat=np.concatenate(lat_all)
print('bagtime-headerstamp latency s: pct 1,50,99,99.9,max', np.percentile(lat,[1,50,99,99.9]), lat.max())
print('out-of-order header stamps', ooo, 'of', tot)
for veh in ('30618','30639'):
    r=[x for x in rows if x[0]==veh]
    kf=np.array([x[2] for x in r]); kr=np.array([x[3] for x in r])
    print(veh, 'n',len(r),'kf median %.5f std %.5f  kr median %.5f std %.5f'%(np.median(kf),kf.std(),np.median(kr),kr.std()), ' 1/3.6=%.5f'%(1/3.6))
    print('  kf range', kf.min(), kf.max(), ' kr range', kr.min(), kr.max())
ad=np.array([x[4] for x in rows]); print('|vf-vr| units pct50/95/99/99.9 median over bags', np.median(ad,axis=0))
