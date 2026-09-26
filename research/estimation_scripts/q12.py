import numpy as np, glob, os
rng=np.random.default_rng(0)
files = sorted(glob.glob('C:/MosTransHack/data/npz/*.npz'))
data=[]
for f in files:
    d=np.load(f); name=os.path.basename(f)[:-4]
    fr=d['vehicle__front_bogie_velocity']; cmd=d['vehicle__driver_position_cmd']
    if len(fr)<600 or len(cmd)<600: continue
    tt=np.arange(fr[0,1]+2, fr[-1,1]-2, 0.1)
    v=np.interp(tt,fr[:,1],fr[:,2])/3.6
    a=np.convolve(np.gradient(v,0.1),np.ones(5)/5,mode='same')
    idx=np.clip(np.searchsorted(cmd[:,1],tt)-1,0,len(cmd)-1); n=cmd[idx,2].astype(int)
    L=3; nl=np.r_[np.full(L,n[0]),n[:-L]]
    data.append((name,v,a,nl))
# fit table with bilinear-ish bins (1 m/s)
X=np.concatenate([d[3] for d in data]); V=np.concatenate([d[1] for d in data]); A=np.concatenate([d[2] for d in data])
m=V>0.3
vb=np.clip(V.astype(int),0,20); key=(X+15)*21+vb
s=np.bincount(key[m],A[m],minlength=31*21); c=np.bincount(key[m],minlength=31*21); tab=s/np.maximum(c,1)
def amod(n,v):
    if v<=0.05 and n<=0: return 0.0
    return tab[(n+15)*21+min(int(max(v,0)),20)]
for T in (1,3,5,10,20,30):
    errs=[]
    for _ in range(3000):
        name,v,a,nl=data[rng.integers(len(data))]
        if len(v)<T*10+10: continue
        i=rng.integers(0,len(v)-T*10-1)
        if v[i:i+T*10].max()<1: continue
        vv=v[i]
        for k in range(i,i+T*10):
            vv=max(0.0,vv+0.1*amod(nl[k],vv))
        errs.append(vv-v[i+T*10])
    e=np.array(errs); print('dead-reckon %2ds: speed err RMSE %.2f m/s  MAE %.2f  p95 |e| %.2f   (n=%d)'%(T,np.sqrt(np.mean(e**2)),np.mean(np.abs(e)),np.percentile(np.abs(e),95),len(e)))
