import numpy as np, glob, os
files = sorted(glob.glob('C:/MosTransHack/data/npz/*.npz'))
iv=[]
for f in files:
    d=np.load(f); fr=d['vehicle__front_bogie_velocity']; name=os.path.basename(f)[:-4]
    if len(fr)<10: continue
    iv.append((name, fr[0,1], fr[-1,1]))
iv.sort(key=lambda x:x[1])
dup=[]; ov=[]
for i in range(len(iv)):
    for j in range(i+1,len(iv)):
        a,b=iv[i],iv[j]
        if b[1]>a[2]: break
        o=min(a[2],b[2])-max(a[1],b[1])
        if o>0: ov.append((a[0],b[0],o, (a[2]-a[1]), (b[2]-b[1])))
print('overlapping pairs (time overlap s):', len(ov))
for x in ov[:30]: print('  %s %s overlap %.0fs len %.0f %.0f'%x)
