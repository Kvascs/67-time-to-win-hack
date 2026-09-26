import sys, pickle; sys.path.insert(0,'C:/MosTransHack/research/map_matching_code')
from common import *
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
SP='C:/MosTransHack/research/map_matching_code/'
M=pickle.load(open(SP+'map.pkl','rb'))['maps']
res={}
for dirn in ('AB','BA'):
    P=M[dirn]['P']; s=np.arange(len(P)).astype(float); tree=cKDTree(P)
    psi=np.unwrap(np.arctan2(*np.gradient(P,axis=0)[:,::-1].T)); kap=gaussian_filter1d(np.gradient(gaussian_filter1d(psi,2)),2)
    V=[]; seen=set()
    for n in names():
        d=load(n); m=d['sensing__gnss__master__fix']
        if m.shape[0]<3000: continue
        key=(m.shape[0],round(m[0,0],1))
        if key in seen: continue
        seen.add(key)
        t,p,z,st=fix_xy(m,True)
        if len(t)<0.9*m.shape[0]: continue
        dd=np.linalg.norm(p[0]-A)<np.linalg.norm(p[0]-B)
        if (dirn=='AB')!=dd: continue
        dist,j=tree.query(p)
        fv=d['vehicle__front_bogie_velocity']; v=np.interp(t,fv[:,1],fv[:,2])
        ok=dist<2
        prof=np.full(len(P)//5+1,np.nan)
        b=j[ok]//5
        for bi in np.unique(b): prof[bi]=np.max(v[ok][b==bi])
        V.append(prof)
    V=np.array(V)
    p50=np.nanmedian(V,0); p90=np.nanpercentile(V,90,axis=0); p10=np.nanpercentile(V,10,axis=0)
    cap=np.nanmax(V,0)
    sb=np.arange(V.shape[1])*5
    print('==',dirn,'runs',len(V))
    # zones where cap is low (speed restriction) : print segments where p90 < 20 km/h and moving
    low=(p90<21)&(p50>3)
    edges=np.diff(np.r_[0,low.astype(int),0]); st_=np.where(edges==1)[0]; en_=np.where(edges==-1)[0]
    for a,c in zip(st_,en_):
        if c-a<3: continue
        print('  restricted zone s=%5d..%5d m (len %4d): p90 speed %.1f km/h, max %.1f, |kappa|max 1/%.0f m'%(sb[a],sb[c-1]+5,(c-a)*5,np.nanmax(p90[a:c]),np.nanmax(cap[a:c]),1/max(np.max(np.abs(kap[sb[a]:sb[c-1]+5])),1e-6)))
    # repeatability of speed at each s (moving portions): std across runs where median>20
    mv=p50>20
    print('  mean across-run std of speed where median>20 km/h: %.1f km/h; median speed %.1f; typical p90-p10 %.1f'%(np.nanmean(np.nanstd(V[:,mv],0)),np.nanmedian(p50[mv]),np.nanmedian((p90-p10)[mv])))
    print('  overall max speed over all runs %.1f km/h; mean of per-bin cap %.1f'%(np.nanmax(cap),np.nanmean(cap[p50>3])))
    res[dirn]=dict(sb=sb,p10=p10,p50=p50,p90=p90,cap=cap)
pickle.dump(res,open(SP+'vprof.pkl','wb'))
