import numpy as np
G=81e9; c11=4.12
Q=45e3                      # wheel load [N] ~ 36.7 t / 8 wheels
r1,r2=0.31,0.30             # wheel radius, rail crown radius [m]
E=210e9; nu=0.28
Es=E/(2*(1-nu**2)); Rs=1/(1/r1+1/r2)
a=(3*Q*Rs/(4*Es))**(1/3); b=a
print(f"Hertz contact a=b={a*1e3:.2f} mm, p0={1.5*Q/(np.pi*a*b)/1e9:.2f} GPa")
params={'dry':dict(kA=1.0,kS=0.4,mu0=0.55,A=0.40,B=0.60),
        'wet':dict(kA=0.30,kS=0.10,mu0=0.30,A=0.40,B=0.20),
        # tram-specific low-adhesion guess: scale wet mu0 to ~0.13 peak (oily/leaves)
        'low(leaves,guess)':dict(kA=0.15,kS=0.05,mu0=0.13,A=0.40,B=0.20)}
def f_polach(s,v,p):
    w=abs(s)*v
    mu=p['mu0']*((1-p['A'])*np.exp(-p['B']*w)+p['A'])
    eps=0.25*G*np.pi*a*b*c11/(Q*mu)*abs(s)
    F=2*mu/np.pi*(p['kA']*eps/(1+(p['kA']*eps)**2)+np.arctan(p['kS']*eps))
    return F
s=np.logspace(-5,np.log10(0.6),4000)
for name,p in params.items():
    for v in (3,8,14,20):
        f=f_polach(s,v,p); i=np.argmax(f)
        # creep needed for mu_util 0.08, 0.12, 0.15
        req={}
        for mu_u in (0.05,0.08,0.12,0.15):
            j=np.where(f[:i+1]>=mu_u)[0]
            req[mu_u]=f"{s[j[0]]*100:.2f}%" if len(j) else "slip"
        print(f"{name:18s} v={v:2d} m/s: f_max={f[i]:.3f} at s={s[i]*100:.2f}% (w={s[i]*v:.3f} m/s); creep for mu_u:",req)
