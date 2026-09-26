import numpy as np, glob, os
files = sorted(glob.glob('C:/MosTransHack/data/npz/*.npz'))
lat0,lon0=55.81,37.46
stops=[]; nb=0; dist=[]
for f in files:
    d=np.load(f); fr=d['vehicle__front_bogie_velocity']
    fx=d['sensing__gnss__master__fix'] if d['sensing__gnss__master__fix'].shape[0]>100 else d['sensing__gnss__rover__fix']
    if len(fr)<100 or len(fx)<100: continue
    nb+=1
    x=(fx[:,3]-lon0)*111320*np.cos(np.radians(lat0)); y=(fx[:,2]-lat0)*110540
    v=fr[:,2]; t=fr[:,1]
    z=(v==0).astype(int); e=np.diff(np.r_[0,z,0]); st=np.where(e==1)[0]; en=np.where(e==-1)[0]
    for s,e2 in zip(st,en):
        if t[e2-1]-t[s]>=3:
            tm=0.5*(t[s]+t[e2-1]); stops.append((np.interp(tm,fx[:,1],x),np.interp(tm,fx[:,1],y)))
    dist.append(np.trapezoid(v/3.6,t))
S=np.array(stops)
print('bags',nb,'stops>=3s',len(S),'per bag',len(S)/nb, ' median dist per bag km', np.median(dist)/1000, ' km per stop', np.sum(dist)/1000/len(S))
# cluster: 15 m grid
from collections import Counter
c=Counter(zip((S[:,0]//15).astype(int),(S[:,1]//15).astype(int)))
cnt=np.array(sorted(c.values(),reverse=True))
print('top 15m-cells stop counts', cnt[:25])
print('fraction of stops in cells with >=10 stops', cnt[cnt>=10].sum()/len(S))
