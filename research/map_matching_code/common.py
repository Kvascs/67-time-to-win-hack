import numpy as np, glob, os
from pyproj import Transformer
LAT0,LON0=55.80,37.47
TR=Transformer.from_crs("EPSG:4326","+proj=tmerc +lat_0=%f +lon_0=%f +k=1 +x_0=0 +y_0=0 +ellps=WGS84"%(LAT0,LON0),always_xy=True)
ROOT='C:/MosTransHack/data/npz/'
A=np.array([-483.,1158.]); B=np.array([-5080.,-50.])
def load(name):
    d=np.load(ROOT+name+'.npz'); return {k:d[k] for k in d.files}
def fix_xy(g, rtk_only=False):
    ok=np.isfinite(g[:,2])&(np.abs(g[:,2])>1)
    if rtk_only: ok&=(g[:,5]==2)
    g=g[ok]; x,y=TR.transform(g[:,3],g[:,2])
    return g[:,0], np.c_[x,y], g[:,4], g[:,5]
def names():
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(ROOT+'*.npz'))
