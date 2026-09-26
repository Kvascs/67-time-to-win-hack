import sys, pickle; sys.path.insert(0,r'C:/MosTransHack/research/map_matching_code')
from common import *
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
SP='C:/MosTransHack/research/map_matching_code/'
def resample(P, ds=1.0, extra=None):
    d=np.r_[0,np.cumsum(np.hypot(*np.diff(P,axis=0).T))]
    keep=np.r_[True,np.diff(d)>1e-6]; P=P[keep]; d=d[keep]
    s=np.arange(0,d[-1],ds)
    out=np.c_[np.interp(s,d,P[:,0]),np.interp(s,d,P[:,1])]
    if extra is not None: return s,out,np.interp(s,d,extra[keep])
    return s,out
class Line:
    def __init__(s_, P, z=None):
        s_.s, s_.P = resample(P)
        s_.z = z
        s_.tree=cKDTree(s_.P)
        t=np.gradient(s_.P,axis=0); t/=np.linalg.norm(t,axis=1)[:,None]; s_.t=t; s_.n=np.c_[-t[:,1],t[:,0]]
    def project_seq(s_, pts, gate=25.0, maxjump=60.0, k=24):
        # fast sequential projection with continuity (KD-tree kNN, then light loop, vectorised refinement)
        n=len(s_.s); N=len(pts)
        S=np.full(N,np.nan); E=np.full(N,np.nan)
        if N==0: return S,E
        d,idx=s_.tree.query(pts,k=k,distance_upper_bound=gate)
        chosen=np.full(N,-1); prev=None; fails=0
        ss=s_.s
        for i in range(N):
            ii=idx[i]; dd=d[i]; v=ii<n
            if not v[0]: continue
            ii=ii[v]; dd=dd[v]
            if prev is not None and fails<5:
                m=np.abs(ss[ii]-prev)<maxjump
                if not m.any():
                    fails+=1; continue
                ii=ii[m]; dd=dd[m]
            j=ii[np.argmin(dd)]; chosen[i]=j; prev=ss[j]; fails=0
        ok=chosen>=0
        j=chosen[ok]; p=pts[ok]
        best_d=np.full(len(j),np.inf); best_s=np.zeros(len(j)); best_e=np.zeros(len(j))
        for a,b in ((np.maximum(j-1,0),j),(j,np.minimum(j+1,n-1))):
            v=s_.P[b]-s_.P[a]; L=np.einsum('ij,ij->i',v,v); L=np.where(L<1e-12,1e-12,L)
            u=np.clip(np.einsum('ij,ij->i',p-s_.P[a],v)/L,0,1)
            q=s_.P[a]+u[:,None]*v; w=p-q; dd=np.hypot(w[:,0],w[:,1])
            sg=np.sign(v[:,0]*w[:,1]-v[:,1]*w[:,0])
            better=(dd<best_d)&(a!=b)
            best_d=np.where(better,dd,best_d); best_s=np.where(better,ss[a]+u*(ss[b]-ss[a]),best_s); best_e=np.where(better,sg*dd,best_e)
        S[ok]=best_s; E[ok]=best_e
        return S,E
# collect runs
runs={'AB':[],'BA':[]}; seen=set()
for n in names():
    d=load(n); m=d['sensing__gnss__master__fix']
    if m.shape[0]<3000: continue
    key=(m.shape[0],round(m[0,0],1))
    if key in seen: continue
    seen.add(key)
    t,p,z,st=fix_xy(m,True)
    if len(t)<0.9*m.shape[0]: continue
    dirn='AB' if np.linalg.norm(p[0]-A)<np.linalg.norm(p[0]-B) else 'BA'
    full=(min(np.linalg.norm(p[0]-A),np.linalg.norm(p[0]-B))<60) and (min(np.linalg.norm(p[-1]-A),np.linalg.norm(p[-1]-B))<150)
    runs[dirn].append(dict(name=n,t=t,p=p,z=z,full=full,veh=n[:5]))
for k in runs: print(k,len(runs[k]),'full',sum(r['full'] for r in runs[k]))
maps={}
for dirn in ('AB','BA'):
    R=[r for r in runs[dirn] if r['full']]
    # reference: longest full run from 30618 with fewest gaps
    S0,S1=(A,B) if dirn=='AB' else (B,A)
    ref=min(R,key=lambda r: np.linalg.norm(r['p'][0]-S0)+np.linalg.norm(r['p'][-1]-S1)+ (0 if len(r['t'])/(r['t'][-1]-r['t'][0])>9.5 else 50))
    print(' ref start/end dist', np.linalg.norm(ref['p'][0]-S0), np.linalg.norm(ref['p'][-1]-S1))
    v=np.r_[0,np.hypot(*np.diff(ref['p'],axis=0).T)]
    mv=np.r_[True, v[1:]>0.05]
    line=Line(ref['p'][mv])
    print(dirn,'ref',ref['name'],'len %.1f'%line.s[-1])
    train=[r for r in R if r['veh']=='30618']
    for it in range(3):
        allS=[];allE=[];allZ=[]
        for r in train:
            S,E=line.project_seq(r['p'])
            ok=np.isfinite(S)&(np.abs(E)<2.5)
            allS.append(S[ok]);allE.append(E[ok]);allZ.append(r['z'][ok])
        S=np.concatenate(allS);E=np.concatenate(allE);Z=np.concatenate(allZ)
        b=np.clip(np.round(S).astype(int),0,len(line.s)-1)
        med=np.zeros(len(line.s)); cnt=np.bincount(b,minlength=len(line.s))
        order=np.argsort(b); bs=b[order]; es=E[order]; zs=Z[order]
        splits=np.searchsorted(bs,np.arange(len(line.s)+1))
        zmed=np.full(len(line.s),np.nan)
        for i in range(len(line.s)):
            a,c=splits[i],splits[i+1]
            if c-a>=3: med[i]=np.median(es[a:c]); zmed[i]=np.median(zs[a:c])
        med=gaussian_filter1d(med,3)
        resid=E-med[b]
        print(' it',it,'pts',len(E),'resid RMS %.3f p95 %.3f p99 %.3f | mean|shift| %.3f'%(np.sqrt(np.mean(resid**2)),np.percentile(np.abs(resid),95),np.percentile(np.abs(resid),99),np.mean(np.abs(med))))
        newP=line.P+med[:,None]*line.n
        okz=np.isfinite(zmed); zf=np.interp(line.s,line.s[okz],zmed[okz])
        zf=gaussian_filter1d(zf,5)
        line=Line(newP); line.z=np.interp(line.s, np.linspace(0,line.s[-1],len(zf)), zf)
    maps[dirn]=dict(s=line.s,P=line.P,z=line.z,ref=ref['name'])
    # test on 30639 runs (cross-vehicle validation of map)
    E2=[]
    for r in R:
        if r['veh']!='30639': continue
        S,E=line.project_seq(r['p']); ok=np.isfinite(S)&(np.abs(E)<5); E2.append(E[ok])
    if E2:
        E2=np.concatenate(E2); print(' 30639 cross-track vs map: RMS %.3f median|e| %.3f p95 %.3f'%(np.sqrt(np.mean(E2**2)),np.median(np.abs(E2)),np.percentile(np.abs(E2),95)))
pickle.dump(dict(maps=maps),open(SP+'map.pkl','wb'))
# geometry stats
for dirn,mp in maps.items():
    P=mp['P']; s=mp['s']
    psi=np.unwrap(np.arctan2(*np.gradient(P,axis=0)[:,::-1].T))
    kap=gaussian_filter1d(np.gradient(psi,s),3)
    gr=np.gradient(gaussian_filter1d(mp['z'],10),s)
    print(dirn,'L=%.1f m  z %.1f..%.1f  max|grade| %.3f  min R %.1f m  total turn %.0f deg  frac |k|>1/500: %.2f'%(s[-1],mp['z'].min(),mp['z'].max(),np.abs(gr).max(),1/np.abs(kap).max(),np.degrees(psi[-1]-psi[0]),np.mean(np.abs(kap)>1/500)))
