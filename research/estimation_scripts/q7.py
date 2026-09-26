import numpy as np, glob, os
files = sorted(glob.glob('C:/MosTransHack/data/npz/*.npz'))
minnz=[]; res=[]; zero_g=[]; stepv=[]
for f in files[:60]:
    d=np.load(f); fr=d['vehicle__front_bogie_velocity']
    gv=d['sensing__gnss__master__vel'] if d['sensing__gnss__master__vel'].shape[0]>100 else d['sensing__gnss__rover__vel']
    if len(fr)<200: continue
    v=fr[:,2]; nz=v[v>0]; 
    if len(nz): minnz.append(nz.min())
    dv=np.abs(np.diff(v)); dv=dv[dv>0]; 
    if len(dv): stepv.append(np.percentile(dv,1))
    # second difference noise estimate at speed>10 km/h
    m=(v[1:-1]>10); d2=(v[2:]-2*v[1:-1]+v[:-2])[m]; res.append(np.std(d2)/np.sqrt(6))
    if len(gv)>100:
        g=np.interp(fr[:,1],gv[:,1],np.hypot(gv[:,2],gv[:,3])); z=(v==0); zero_g.append(g[z])
Z=np.concatenate(zero_g)
print('min nonzero wheel reading km/h (pct of bags 0/50/100):', np.percentile(minnz,[0,50,100]))
print('smallest nonzero step km/h pct:', np.percentile(stepv,[0,50,100]))
print('white-noise sigma est from 2nd diff (km/h): median', np.nanmedian(res), ' -> m/s', np.nanmedian(res)/3.6)
print('GNSS speed when wheel==0: pct 50/99/99.9', np.percentile(Z,[50,99,99.9]), ' frac>0.3m/s', np.mean(Z>0.3))
