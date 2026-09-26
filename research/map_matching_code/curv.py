import sys, pickle; sys.path.insert(0,'C:/MosTransHack/research/map_matching_code')
from common import *
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
SP='C:/MosTransHack/research/map_matching_code/'
M=pickle.load(open(SP+'map.pkl','rb'))['maps']
geo={}
for k,v in M.items():
    P=v['P']; s=v['s'][:len(P)]
    psi=np.unwrap(np.arctan2(*np.gradient(P,axis=0)[:,::-1].T))
    kap=gaussian_filter1d(np.gradient(gaussian_filter1d(psi,2),s),2)
    geo[k]=(cKDTree(P),s,kap)
X=[];seen=set()
for n in names():
    d=load(n); m=d['sensing__gnss__master__fix']
    if m.shape[0]<3000: continue
    key=(m.shape[0],round(m[0,0],1))
    if key in seen: continue
    seen.add(key)
    t,p,z,st=fix_xy(m,True)
    if len(t)<0.9*m.shape[0]: continue
    dirn='AB' if np.linalg.norm(p[0]-A)<np.linalg.norm(p[0]-B) else 'BA'
    tree,s,kap=geo[dirn]; dd,j=tree.query(p)
    fv=d['vehicle__front_bogie_velocity']; rv=d['vehicle__rear_bogie_velocity']; mv=d['sensing__gnss__master__vel']
    # resample all to master fix times
    vf=np.interp(t,fv[:,1],fv[:,2])/3.6; vr=np.interp(t,rv[:,1],rv[:,2])/3.6
    gs=np.hypot(np.interp(t,mv[:,1],mv[:,2]),np.interp(t,mv[:,1],mv[:,3]))
    k_=kap[j]
    ok=(dd<1.5)&(gs>3)
    X.append(np.c_[k_[ok],vf[ok],vr[ok],gs[ok],np.full(ok.sum(),n.startswith('30639'))])
X=np.vstack(X); k_,vf,vr,gs,veh=X.T
ak=np.abs(k_)
diff=(vf-vr)/(0.5*(vf+vr))
rat=(0.5*(vf+vr))/gs
print('N',len(X))
bins=[0,1/2000,1/500,1/200,1/100,1/50,1/25,1]
for a,b in zip(bins[:-1],bins[1:]):
    m=(ak>=a)&(ak<b)
    if m.sum()<50: continue
    print('R in (%6.0f,%6.0f] m: n=%6d  (vf-vr)/v mean %+.4f std %.4f | wheel/GNSS mean %.4f  front/GNSS %.4f rear/GNSS %.4f'%(1/b if b<1 else 1,1/a if a>0 else np.inf,m.sum(),diff[m].mean(),diff[m].std(),rat[m].mean(),(vf/gs)[m].mean(),(vr/gs)[m].mean()))
# signed curvature correlation
for sgn,lab in ((1,'left(k>0)'),(-1,'right(k<0)')):
    m=(sgn*k_>1/100)
    print(lab,'n',m.sum(),'(vf-vr)/v mean %+.4f'%diff[m].mean())
print('corr(diff, |k|)=%.3f corr(diff,k)=%.3f corr(rat,|k|)=%.3f'%(np.corrcoef(diff,ak)[0,1],np.corrcoef(diff,k_)[0,1],np.corrcoef(rat,ak)[0,1]))
