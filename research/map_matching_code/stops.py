import sys, pickle; sys.path.insert(0,'C:/MosTransHack/research/map_matching_code')
from common import *
exec(open('C:/MosTransHack/research/map_matching_code/buildmap.py').read().split('# collect runs')[0])
M=pickle.load(open(SP+'map.pkl','rb'))['maps']
lines={k:Line(v['P']) for k,v in M.items()}
for k in lines: lines[k].z=M[k]['z']
seen=set(); ev=[]; runinfo=[]
for n in names():
    d=load(n); m=d['sensing__gnss__master__fix']
    if m.shape[0]<3000: continue
    key=(m.shape[0],round(m[0,0],1))
    if key in seen: continue
    seen.add(key)
    t,p,z,st=fix_xy(m,True)
    if len(t)<0.9*m.shape[0]: continue
    dirn='AB' if np.linalg.norm(p[0]-A)<np.linalg.norm(p[0]-B) else 'BA'
    L=lines[dirn]; S,E=L.project_seq(p)
    fv=d['vehicle__front_bogie_velocity']; rv=d['vehicle__rear_bogie_velocity']
    tw=fv[:,1]; vw=0.5*(fv[:,2]+np.interp(tw,rv[:,1],rv[:,2]))
    # stop events: wheel speed == 0 for >= 2s
    z0=vw<0.05
    edges=np.diff(np.r_[0,z0.astype(int),0]); st_i=np.where(edges==1)[0]; en_i=np.where(edges==-1)[0]-1
    ok=np.isfinite(S)
    for a,b in zip(st_i,en_i):
        dur=tw[b]-tw[a]
        if dur<2.0: continue
        tm=0.5*(tw[a]+tw[b])
        sg=np.interp(tm,t[ok],S[ok])
        ev.append((n,dirn,sg,dur,tw[a]-tw[0]))
    runinfo.append((n,dirn))
ev_arr=np.array([(e[2],e[3]) for e in ev]); dirs=np.array([e[1] for e in ev]); rn=np.array([e[0] for e in ev])
pickle.dump(dict(ev=ev,runinfo=runinfo),open(SP+'stops.pkl','wb'))
for dirn in ('AB','BA'):
    nr=sum(1 for r in runinfo if r[1]==dirn)
    sel=dirs==dirn; s=ev_arr[sel,0]; du=ev_arr[sel,1]; r_=rn[sel]
    o=np.argsort(s); s=s[o]; du=du[o]; r_=r_[o]
    # 1D clustering with gap 15 m
    cl=np.r_[0,np.cumsum(np.diff(s)>15)]
    print(f'== {dirn}: runs {nr}, stop events {len(s)}')
    for c in np.unique(cl):
        m=cl==c
        nrun=len(set(r_[m]))
        if nrun<3: continue
        print('  s=%7.1f  n=%3d runs=%2d (%.0f%%)  std=%5.2f  IQR=%5.2f  dwell med %5.1fs p10 %5.1f p90 %5.1f'%(np.median(s[m]),m.sum(),nrun,100*nrun/nr,np.std(s[m]),np.subtract(*np.percentile(s[m],[75,25])),np.median(du[m]),np.percentile(du[m],10),np.percentile(du[m],90)))
    print('  singletons/rare clusters events:',sum(1 for c in np.unique(cl) if len(set(r_[cl==c]))<3))
