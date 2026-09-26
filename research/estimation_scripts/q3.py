import numpy as np, glob, os
files = sorted(glob.glob('C:/MosTransHack/data/npz/*.npz'))
lags=np.arange(-0.6,0.61,0.02)
acc_rmse=np.zeros(len(lags)); cnt=0
glat=[]
best=[]
for f in files:
    d=np.load(f); name=os.path.basename(f)[:-4]
    fr=d['vehicle__front_bogie_velocity']
    gv=d['sensing__gnss__master__vel'] if d['sensing__gnss__master__vel'].shape[0]>100 else d['sensing__gnss__rover__vel']
    gf=d['sensing__gnss__master__fix']
    if len(gf)>10: glat.append(gf[:,0]-gf[:,1])
    if len(gv)>10: glat.append(gv[:,0]-gv[:,1])
    if len(fr)<1000 or len(gv)<1000: continue
    k=0.278 if name.startswith('30618') else 0.2763
    tg=gv[:,1]; g=np.hypot(gv[:,2],gv[:,3])
    tt=np.arange(max(tg[0],fr[0,1])+1, min(tg[-1],fr[-1,1])-1, 0.05)
    G=np.interp(tt,tg,g)
    r=[]
    for L in lags:
        W=k*np.interp(tt+L,fr[:,1],fr[:,2])   # wheel at time t+L compared with gnss at t
        m=G>0.5
        r.append(np.sqrt(np.mean((W[m]-G[m])**2)))
    r=np.array(r); acc_rmse+=r; cnt+=1; best.append(lags[np.argmin(r)])
G=np.concatenate(glat)
print('GNSS bag-header latency pct50/99/max', np.percentile(G,[50,99]), G.max())
print('best lag (wheel_time = gnss_time + L) median/pct10/pct90', np.median(best), np.percentile(best,[10,90]))
i=np.argmin(acc_rmse); print('pooled best lag', lags[i], 'rmse at best %.3f at 0 %.3f'%(acc_rmse[i]/cnt, acc_rmse[np.argmin(abs(lags))]/cnt))
