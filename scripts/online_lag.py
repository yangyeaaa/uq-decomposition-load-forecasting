from common import *
out=open(f"{OUT}/online_lag_log.txt","w")
def log(x): print(x,flush=True); out.write(x+"\n"); out.flush()
def qhat(s,c): n=len(s); k=min(n,int(np.ceil((n+1)*c))); return np.sort(s)[k-1]
def run(name):
    d=prep_daystrat(name); sc,mn=d["sdl"].scale_[0],d["sdl"].mean_[0]
    P=np.load(f"{OUT}/{name}_ensK10_preds.npz"); f=np.sqrt(10/9)
    mu,ale,epi=P["mu"],P["ale"],P["epi"]*f; muc,alec,epic=P["muc"],P["alec"],P["epic"]*f; y,yc=P["y"],P["yc"]
    tot=np.sqrt(ale**2+epi**2); totc=np.sqrt(alec**2+epic**2)
    Y=y*sc+mn; M=mu*sc+mn; S=tot*sc; s_cal=np.abs(yc-muc)/totc; s_te=np.abs(y-mu)/tot
    N=len(y); days=[np.arange(i,min(i+24,N)) for i in range(0,N,24)]; z=norm.ppf((1+LEVELS)/2)
    tau=np.sqrt(np.mean(s_cal**2))
    log(f"\n=== {name}: K=10 ensemble, online calibration with one-day lag (multiplier for day d+1 uses scores through day d-1) ===")
    for lab,spec in [("Raw (Gaussian)",None),("Global sd scaling",("tau",None)),("Static split conformal",("static",None)),("Rolling W=30d (lagged)",("roll",30)),("Rolling W=90d (lagged)",("roll",90)),("ACI gamma=0.01 on W=30d (lagged)",("aci",30))]:
        cov=np.zeros((N,len(LEVELS))); w90=np.zeros(N); alpha=1-LEVELS.copy(); hist=list(s_cal); pending=None
        for idx in days:
            if spec is None: mult=z
            elif spec[0]=="tau": mult=z*tau
            elif spec[0]=="static": mult=np.array([qhat(np.array(s_cal),c) for c in LEVELS])
            else:
                pool=np.array(hist[-spec[1]*24:])
                if spec[0]=="roll": mult=np.array([qhat(pool,c) for c in LEVELS])
                else: mult=np.array([qhat(pool,min(max(1-a,0.01),0.999)) for a in alpha])
            for j in range(len(LEVELS)): cov[idx,j]=s_te[idx]<=mult[j]
            w90[idx]=2*mult[-2]*S[idx]
            # scores of day d become available only for day d+2: append the previous day's pending scores now
            if pending is not None:
                hist+=list(pending["s"])
                if spec is not None and spec[0]=="aci": alpha=alpha+0.01*((1-LEVELS)-(1-pending["cov"].mean(0)))
            pending=dict(s=s_te[idx],cov=cov[idx])
        obs=cov.mean(0); lo,hi=M-w90/2,M+w90/2; a=0.1
        log(f"{lab:34s} ECE={np.mean(np.abs(LEVELS-obs)):.4f} PICP90={obs[-2]:.4f} IS90={np.mean((hi-lo)+2/a*(lo-Y)*(Y<lo)+2/a*(Y-hi)*(Y>hi)):.5g}")
for n in ["GEF","COVID"]: run(n)
log("DONE")
