import numpy as np
pairs=[('30618_79b204dc','30618_887a2b9a'),('30618_2161b58b','30618_f28179bb'),('30618_4d487b0d','30618_b3042f78'),('30618_40ffd323','30618_efb92709'),('30618_2cb9ce37','30618_5eb8d2c9'),('30618_0259fe53','30618_cfd9fd5a')]
for a,b in pairs:
    A=np.load(f'C:/MosTransHack/data/npz/{a}.npz'); B=np.load(f'C:/MosTransHack/data/npz/{b}.npz')
    out=[a,b]
    for k in A.files:
        x=A[k]; y=B[k]
        if x.shape!=y.shape:
            out.append(f'{k}: shape {x.shape} vs {y.shape}'); continue
        if len(x)==0: continue
        dd=np.abs(x[:,2:]-y[:,2:]).max() if x.shape[1]>2 else 0
        dt=np.abs(x[:,:2]-y[:,:2]).max()
        if dd>0 or dt>0: out.append(f'{k}: maxdiff val {dd:.4g} time {dt:.4g}')
    print(' | '.join(out))
