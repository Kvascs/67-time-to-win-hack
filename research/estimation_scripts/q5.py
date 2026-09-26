import numpy as np, glob, os
files = sorted(glob.glob('C:/MosTransHack/data/npz/*.npz'))
tot=0; zero_mis=0; spikes=0; bagsz=[]; accs=[]
for f in files:
    d=np.load(f); name=os.path.basename(f)[:-4]
    fr=d['vehicle__front_bogie_velocity']; rr=d['vehicle__rear_bogie_velocity']
    if len(fr)<100 or len(rr)<100: continue
    t=fr[:,1]; vf=fr[:,2]; vr=np.interp(t,rr[:,1],rr[:,2])
    z=((vf>5)&(vr<0.5))|((vr>5)&(vf<0.5)); zero_mis+=z.sum(); tot+=len(t)
    if z.sum()>0: bagsz.append((name,z.sum()))
    # spikes: deviation from 5-pt median
    from numpy.lib.stride_tricks import sliding_window_view as sw
    if len(vf)>5:
        med=np.median(sw(vf,5),axis=1); dev=np.abs(vf[2:-2]-med); spikes+=np.sum(dev>2.0)
    # wheel-derived acceleration (m/s^2), 1 s central diff
    v=vf/3.6; 
    if len(v)>20:
        a=(v[10:]-v[:-10])/(t[10:]-t[:-10]); accs.append(a[np.isfinite(a)])
A=np.concatenate(accs)
print('zero-vs-moving mismatch samples', zero_mis, 'of', tot, 'bags:', bagsz[:10])
print('single-sample spikes >2 km/h from 5pt median:', spikes)
print('wheel accel pct 0.1/1/50/99/99.9', np.percentile(A,[0.1,1,50,99,99.9]))
