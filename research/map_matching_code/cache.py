import sys, pickle; sys.path.insert(0,'C:/MosTransHack/research/map_matching_code')
from common import *
exec(open('C:/MosTransHack/research/map_matching_code/buildmap.py').read().split('# collect runs')[0])
M=pickle.load(open(SP+'map.pkl','rb'))['maps']
lines={k:Line(v['P']) for k,v in M.items()}
C={}; seen=set()
for n in names():
    d=load(n); m=d['sensing__gnss__master__fix']
    if m.shape[0]<3000: continue
    key=(m.shape[0],round(m[0,0],1))
    if key in seen: continue
    seen.add(key)
    t,p,z,st=fix_xy(m,True)
    if len(t)<0.9*m.shape[0]: continue
    dirn='AB' if np.linalg.norm(p[0]-A)<np.linalg.norm(p[0]-B) else 'BA'
    Sg,Eg=lines[dirn].project_seq(p)
    fv=d['vehicle__front_bogie_velocity']; rv=d['vehicle__rear_bogie_velocity']
    tw=fv[:,1]; vf=fv[:,2]; vr=np.interp(tw,rv[:,1],rv[:,2])
    tall,pall,zall,stall=fix_xy(m,False)
    C[n]=dict(dir=dirn,t=t,p=p,z=z,Sg=Sg,Eg=Eg,tw=tw,vf=vf,vr=vr,tall=tall,pall=pall,zall=zall,stall=stall)
    print(n,dirn,len(t),flush=True)
pickle.dump(C,open(SP+'cache.pkl','wb'))
print('done')
