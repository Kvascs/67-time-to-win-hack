import sys, pickle; sys.path.insert(0,'C:/MosTransHack/research/map_matching_code')
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SP='C:/MosTransHack/research/map_matching_code/'
M=pickle.load(open(SP+'map.pkl','rb'))['maps']
fig,ax=plt.subplots(1,2,figsize=(16,8))
seen=set()
for n in names():
    d=load(n); m=d['sensing__gnss__master__fix']
    if m.shape[0]<3000: continue
    t,p,z,st=fix_xy(m,True)
    if len(t)<100: continue
    dirn='AB' if np.linalg.norm(p[0]-A)<np.linalg.norm(p[0]-B) else 'BA'
    c='C0' if dirn=='AB' else 'C3'
    for a,(cx,cy) in zip(ax,[(-500,1120),(-5050,-20)]):
        a.plot(p[:,0],p[:,1],'.',ms=1,color=c,alpha=0.3)
for a,(cx,cy) in zip(ax,[(-500,1120),(-5050,-20)]):
    for k,c in (('AB','b'),('BA','r')):
        P=M[k]['P']; a.plot(P[:,0],P[:,1],'-',color=c,lw=1.5,label=k); a.plot(P[0,0],P[0,1],'o',color=c); a.plot(P[-1,0],P[-1,1],'s',color=c)
    a.set_xlim(cx-150,cx+150); a.set_ylim(cy-150,cy+150); a.set_aspect('equal'); a.grid(); a.legend()
plt.savefig(SP+'loops.png',dpi=70)
