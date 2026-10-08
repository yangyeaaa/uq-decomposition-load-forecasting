from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def qhat(s,c): n=len(s); k=min(n,int(np.ceil((n+1)*c))); return np.sort(s)[k-1]
def compute(name):
    d=prep_daystrat(name); sc,mn=d["sdl"].scale_[0],d["sdl"].mean_[0]
    P=np.load(f"{OUT}/{name}_ensK10_preds.npz"); f=np.sqrt(10/9)
    mu,ale,epi=P["mu"],P["ale"],P["epi"]*f; muc,alec,epic=P["muc"],P["alec"],P["epic"]*f; y,yc=P["y"],P["yc"]
    tot=np.sqrt(ale**2+epi**2); totc=np.sqrt(alec**2+epic**2); s_cal=np.abs(yc-muc)/totc; s_te=np.abs(y-mu)/tot
    N=len(y); days=[np.arange(i,min(i+24,N)) for i in range(0,N,24)]; z=norm.ppf((1+LEVELS)/2); tau=np.sqrt(np.mean(s_cal**2))
    res={}
    for lab,spec in [("Raw",None),("Global SD scaling",("tau",None)),("Static split conformal",("static",None)),("Rolling conformal, W=90 d",("roll",90)),("ACI-style, W=30 d",("aci",30))]:
        cov=np.zeros((N,len(LEVELS))); alpha=1-LEVELS.copy(); hist=list(s_cal); pending=None
        for idx in days:
            if spec is None: mult=z
            elif spec[0]=="tau": mult=z*tau
            elif spec[0]=="static": mult=np.array([qhat(np.array(s_cal),c) for c in LEVELS])
            else:
                pool=np.array(hist[-spec[1]*24:])
                mult=np.array([qhat(pool,c) for c in LEVELS]) if spec[0]=="roll" else np.array([qhat(pool,min(max(1-a,0.01),len(pool)/(len(pool)+1))) for a in alpha])
            for j in range(len(LEVELS)): cov[idx,j]=s_te[idx]<=mult[j]
            if pending is not None:
                hist+=list(pending["s"])
                if spec is not None and spec[0]=="aci": alpha=alpha+0.01*((1-LEVELS)-(1-pending["cov"].mean(0)))
            pending=dict(s=s_te[idx],cov=cov[idx])
        c90=cov[:,-2]; nd=len(days); trail=[np.mean(c90[(i-29)*24:(i+1)*24]) for i in range(29,nd)]
        res[lab]=dict(curve=cov.mean(0),trail=np.array(trail))
    return res
R={n:compute(n) for n in ["GEF","COVID"]}
B,O,G,P,R_="#3b6aa5","#c27a3a","#3a8f6e","#7a5aa8","#c0392b"
fig,axes=plt.subplots(2,2,figsize=(11,8.2))
for j,(name,tag) in enumerate([("GEF","GEFCom2014"),("COVID","post-COVID")]):
    ax=axes[0,j]; r=R[name]
    ax.fill_between([0.5,0.95],[0.45,0.90],[0.55,1.0],color="0.9",label="±0.05 tolerance band"); ax.plot([0.5,0.95],[0.5,0.95],"k--",lw=1,label="Ideal")
    ax.plot(LEVELS,r["Raw"]["curve"],"o-",color="0.45",label="Raw"); ax.plot(LEVELS,r["Global SD scaling"]["curve"],"^--",color=O,label="Global SD scaling")
    ax.plot(LEVELS,r["Static split conformal"]["curve"],"s-",color=G,label="Static split conformal"); ax.plot(LEVELS,r["Rolling conformal, W=90 d"]["curve"],"D-",color=B,label="Rolling conformal, W=90 d")
    ax.set_xlim(0.48,0.97); ax.set_ylim(0.43,1.0); ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage"); ax.set_title(f"({'ab'[j]}) {tag}: coverage by level",loc="left",fontweight="bold"); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8,loc="upper left")
    ax=axes[1,j]
    for lab,c,ls in [("Raw","0.45","-"),("Static split conformal",G,"-"),("Rolling conformal, W=90 d",B,"-"),("ACI-style, W=30 d",R_,"--")]:
        ax.plot(np.arange(29,29+len(r[lab]["trail"])),r[lab]["trail"],color=c,ls=ls,lw=1.5,label=lab)
    ax.axhline(0.9,color="k",ls=":",lw=1); ax.set_ylim(0.75,1.0); ax.set_xlabel("Test day"); ax.set_ylabel("Trailing 30-day coverage, 90% interval")
    ax.set_title(f"({'cd'[j]}) {tag}: trailing 30-day coverage",loc="left",fontweight="bold"); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8,loc="lower left")
plt.tight_layout()
for ext in ["pdf","png"]: plt.savefig(f"fig5_calibration.{ext}",dpi=220 if ext=="png" else None)
print("ok")
