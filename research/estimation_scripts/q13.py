exec(open(__file__.replace('q13.py','q12.py')).read().split('for T in')[0])
for T in (3,10,30):
    res={'none':[], 'b5':[], 'b5decay':[]}
    rs=np.random.default_rng(1)
    for _ in range(3000):
        name,v,a,nl=data[rs.integers(len(data))]
        if len(v)<T*10+80: continue
        i=rs.integers(60,len(v)-T*10-1)
        if v[i:i+T*10].max()<1: continue
        pre=[a[k]-amod(nl[k],v[k]) for k in range(i-50,i) if v[k]>0.3]
        b=np.mean(pre) if len(pre)>10 else 0.0
        for mode in res:
            vv=v[i]; bb=b if mode!='none' else 0.0
            for k in range(i,i+T*10):
                vv=max(0.0,vv+0.1*(amod(nl[k],vv)+bb))
                if mode=='b5decay': bb*=np.exp(-0.1/10.0)
            res[mode].append(vv-v[i+T*10])
    print(T,'s', {k:round(float(np.sqrt(np.mean(np.array(e)**2))),3) for k,e in res.items()})
