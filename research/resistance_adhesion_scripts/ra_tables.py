import numpy as np
print("Adhesion vs speed")
for v in (0,10,20,30,40,50,60,70):
    ck=7.5/(v+44)+0.161; ko=9/(v+42)+0.116
    print(f"v={v:3d} km/h  CK={ck:.3f}  Kother={ko:.3f}")
print("\nTakaoka-Kawamura mu(vs)=c e^{-a vs} - d e^{-b vs}")
for name,(aa,ba,ca,da) in {'dry':(0.54,1.2,1,1),'slightly wet':(0.54,1.2,0.2,0.2),'wet':(0.05,0.5,0.08,0.08)}.items():
    vs=np.linspace(0,20,200001); mu=ca*np.exp(-aa*vs)-da*np.exp(-ba*vs); i=np.argmax(mu)
    print(f"{name:13s} mu_max={mu[i]:.3f} at vs={vs[i]:.2f} m/s; slope at 0 = {ca*(-aa)-da*(-ba):.3f} 1/(m/s)")
print("\nCurve resistance N/kN")
print(" R   Rockl(std)  SP98 500/R  PTR 700/R  Protopapadakis mu=0.2 (s=1.59,a=1.8)")
for R in (20,25,30,50,75,100,150,200,300,500):
    rk= (500/(R-30) if R<300 else 650/(R-55)) if R>30 else 0
    rk= f"{rk:6.1f}" if R>30 else "  n/a "
    pp=0.2*(0.72*1.59+0.47*1.8)/R*1000
    print(f"{R:4d} {rk}      {500/R:6.1f}     {700/R:6.1f}     {pp:6.1f}")
print("\nResistance comparisons (N/kN), m=30 t (g=9.81)")
m=30000; W=m*9.81
def t3(v): return (0.0147*m+125.83*v)/W*1000
def prior(v): return (3.0*W/1000 + 60*v + 4.3*v*v)/W*1000
def lit_emu(v): return (1.839*m + 0.0036*m*v + 4.329*v*v)/W*1000  # Rochard-Schmid EMU form as given in Do et al (units uncertain)
for vk in (0,10,20,30,40,50,60,70):
    v=vk/3.6
    print(f"v={vk:2d} km/h  T3-identified={t3(v):5.2f}  prior(A=3N/kN,B=60,C=4.3)={prior(v):5.2f}")
print("\nDeceleration equivalents: 1 N/kN -> a = 9.81e-3/(1+gamma) m/s^2 =", 9.81e-3/1.08)
# rotating mass estimate
i=6.921; r=0.30
for Jm in (0.4,0.7,1.0):
    for Jws in (15,25):
        mrot=4*Jm*i*i/r**2 + 4*Jws/r**2
        print(f"Jmotor-side={Jm} kg m2, Jwheelset={Jws} kg m2 -> m_rot={mrot/1000:.2f} t; gamma(24t)={mrot/24000:.3f}; gamma(34t)={mrot/34000:.3f}")
# traction envelope prior
F0=37.7e3; P=288e3*0.93
for vk in (10,20,27,30,40,50,60,70):
    v=vk/3.6; F=min(F0,P/v)
    print(f"v={vk} km/h F_max~{F/1000:.1f} kN a_empty~{(F-prior(v)*24*9.81)/26200:.2f} m/s2 a_full~{(F-prior(v)*34.5*9.81)/36700:.2f} mu_dem_empty={F/(24000*9.81):.3f}")
