import numpy as np, glob, os
files = sorted(glob.glob('C:/MosTransHack/data/npz/*.npz'))
data=[]
for f in files:
    d=np.load(f); name=os.path.basename(f)[:-4]
    fr=d['vehicle__front_bogie_velocity']; cmd=d['vehicle__driver_position_cmd']
    if len(fr)<600 or len(cmd)<600: continue
    tt=np.arange(fr[0,1]+2, fr[-1,1]-2, 0.1)
    v=np.interp(tt,fr[:,1],fr[:,2])/3.6
    a=np.gradient(v,0.1); 
    # smooth a with 0.5s box
    k=np.ones(5)/5; a=np.convolve(a,k,mode='same')
    idx=np.clip(np.searchsorted(cmd[:,1],tt)-1,0,len(cmd)-1); n=cmd[idx,2]
    data.append((name,tt,v,a,n))
# lag scan: correlate a(t) with table(n(t-L), v)
def table_fit(lagsteps):
    X=[];Y=[]
    for name,tt,v,a,n in data:
        nl=np.r_[np.full(lagsteps,n[0]),n[:len(n)-lagsteps]] if lagsteps>0 else n
        m=v>0.5
        X.append(np.c_[nl[m],v[m]]); Y.append(a[m])
    X=np.vstack(X); Y=np.concatenate(Y)
    vb=np.clip((X[:,1]/2).astype(int),0,10)  # 2 m/s bins
    key=(X[:,0].astype(int)+15)*11+vb
    s=np.bincount(key,Y,minlength=31*11); c=np.bincount(key,minlength=31*11)
    mu=s/np.maximum(c,1); pred=mu[key]
    return np.sqrt(np.mean((Y-pred)**2)), np.std(Y), mu.reshape(31,11), c.reshape(31,11)
for L in [0,3,5,8,10,15,20]:
    r,sd,mu,c=table_fit(L); print('lag %.1fs table-residual rmse %.3f (std a %.3f)'%(L/10,r,sd))
r,sd,mu,c=table_fit(8)
print('mean accel by notch (rows) at v-bins 0-2,2-4,...(m/s):')
for i in range(31):
    if c[i].sum()>500: print('%3d'%(i-15),' '.join('%+.2f'%mu[i,j] if c[i,j]>100 else '  -  ' for j in range(9)))
