import numpy as np, glob, os
files = sorted(glob.glob('C:/MosTransHack/data/npz/*.npz'))
lat0,lon0=55.81,37.46
cells={}; tracks={}
for f in files:
    d=np.load(f); name=os.path.basename(f)[:-4]
    fx=d['sensing__gnss__master__fix'] if d['sensing__gnss__master__fix'].shape[0]>100 else d['sensing__gnss__rover__fix']
    if len(fx)<100: continue
    ok=fx[:,5]>=0
    x=(fx[ok,3]-lon0)*111320*np.cos(np.radians(lat0)); y=(fx[ok,2]-lat0)*110540
    c=set(zip((x//25).astype(int),(y//25).astype(int)))
    tracks[name]=c
    for cc in c: cells[cc]=cells.get(cc,0)+1
cov=[]
for n,c in tracks.items():
    k=[cells[cc]-1 for cc in c]
    cov.append(np.mean(np.array(k)>=2))
cov=np.array(cov)
print('bags', len(tracks), 'unique 25m cells', len(cells))
print('fraction of each bag cells visited by >=2 other bags: pct 0/10/50/90', np.percentile(cov,[0,10,50,90]))
xs=np.array([k[0] for k in cells]); ys=np.array([k[1] for k in cells])
print('extent km x', (xs.max()-xs.min())*25/1000, 'y', (ys.max()-ys.min())*25/1000)
