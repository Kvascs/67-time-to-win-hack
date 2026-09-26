import sys, pickle; sys.path.insert(0,'C:/MosTransHack/research/map_matching_code')
from common import *
from scipy.spatial import cKDTree
SP='C:/MosTransHack/research/map_matching_code/'
M=pickle.load(open(SP+'map.pkl','rb'))['maps']; VP=pickle.load(open(SP+'vprof.pkl','rb'))
for dirn in ('AB','BA'):
    P=M[dirn]['P']; tree=cKDTree(P); vp=VP[dirn]
    low=(vp['p90']<21)&(vp['p50']>3); sb=vp['sb']
    edges=np.diff(np.r_[0,low.astype(int),0]); st_=np.where(edges==1)[0]; en_=np.where(edges==-1)[0]
    zones=[(sb[a],sb[c-1]+5) for a,c in zip(st_,en_) if c-a>=3]
    ev_up={z:[] for z in zones}; ev_dn={z:[] for z in zones}
    seen=set()
    for n in names():
        d=load(n); m=d['sensing__gnss__master__fix']
        if m.shape[0]<3000: continue
        key=(m.shape[0],round(m[0,0],1))
        if key in seen: continue
        seen.add(key)
        t,p,z,st=fix_xy(m,True)
        if len(t)<0.9*m.shape[0]: continue
        if (dirn=='AB')!=(np.linalg.norm(p[0]-A)<np.linalg.norm(p[0]-B)): continue
        dist,j=tree.query(p); ok=dist<2
        fv=d['vehicle__front_bogie_velocity']; v=np.interp(t,fv[:,1],fv[:,2])
        s=j[ok].astype(float); v=v[ok]
        for (a,b) in zones:
            # upward crossing of 22 km/h after zone end region: first time s in [b-20,b+120] where v crosses 22 upward
            w=(s>b-30)&(s<b+150)
            idx=np.where(w)[0]
            if len(idx)<5: continue
            vv=v[idx]; ss=s[idx]
            cr=np.where((vv[:-1]<22)&(vv[1:]>=22))[0]
            if len(cr): ev_up[(a,b)].append(ss[cr[0]])
            # downward crossing of 22 km/h before zone start
            w=(s>a-150)&(s<a+30); idx=np.where(w)[0]
            if len(idx)<5: continue
            vv=v[idx]; ss=s[idx]
            cr=np.where((vv[:-1]>=22)&(vv[1:]<22))[0]
            if len(cr): ev_dn[(a,b)].append(ss[cr[-1]])
    print('==',dirn)
    for zz in zones:
        u=np.array(ev_up[zz]); dn=np.array(ev_dn[zz])
        f=lambda x: (len(x), np.median(x) if len(x) else np.nan, 1.4826*np.median(np.abs(x-np.median(x))) if len(x) else np.nan)
        print('  zone %5d..%5d  exit-accel(22km/h up): n=%2d med %7.1f MAD-sigma %5.1f | entry-decel(22 down): n=%2d med %7.1f MAD-sigma %5.1f'%(zz[0],zz[1],*f(u),*f(dn)))
