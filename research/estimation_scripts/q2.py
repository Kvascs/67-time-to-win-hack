import numpy as np, glob, os
files = sorted(glob.glob('C:/MosTransHack/data/npz/*.npz'))
lat={'f':[], 'r':[], 'c':[]}; dts={'f':[], 'r':[], 'c':[]}
err_by_notch={}; big=[]; stuck=0
for f in files:
    d=np.load(f); name=os.path.basename(f)[:-4]
    fr=d['vehicle__front_bogie_velocity']; rr=d['vehicle__rear_bogie_velocity']; cmd=d['vehicle__driver_position_cmd']
    for k,a in (('f',fr),('r',rr),('c',cmd)):
        if len(a)>10: lat[k].append(a[:,0]-a[:,1]); dts[k].append(np.diff(a[:,1]))
    gv=d['sensing__gnss__master__vel'] if d['sensing__gnss__master__vel'].shape[0]>100 else d['sensing__gnss__rover__vel']
    if len(fr)<100 or len(gv)<100 or len(cmd)<100: continue
    k=0.278 if name.startswith('30618') else 0.2763
    t=fr[:,1]; g=np.interp(t,gv[:,1],np.hypot(gv[:,2],gv[:,3])); v=k*fr[:,2]
    # notch: last command before t
    idx=np.searchsorted(cmd[:,1],t)-1; idx=np.clip(idx,0,len(cmd)-1); n=cmd[idx,2]
    e=v-g
    for nn in np.unique(n):
        err_by_notch.setdefault(int(nn),[]).append(e[n==nn])
    # episodes where |e|>1 m/s
    big.append(np.mean(np.abs(e)>1.0))
for k in lat:
    L=np.concatenate(lat[k]); D=np.concatenate(dts[k])
    print(k,'latency pct50/99/99.9/max %.4f %.4f %.3f %.2f'%tuple(np.r_[np.percentile(L,[50,99,99.9]),L.max()]),
          ' dt pct1/50/99/max %.4f %.4f %.4f %.2f'%tuple(np.r_[np.percentile(D,[1,50,99]),D.max()]), 'frac dt>0.3s %.5f'%np.mean(D>0.3))
print('frac |v_wheel-v_gnss|>1 m/s per bag median/max', np.median(big), np.max(big))
for nn in sorted(err_by_notch):
    e=np.concatenate(err_by_notch[nn])
    if len(e)<200: continue
    print('notch %3d n=%7d  bias %+.3f  std %.3f  p1 %+.2f p99 %+.2f'%(nn,len(e),np.median(e),e.std(),*np.percentile(e,[1,99])))
