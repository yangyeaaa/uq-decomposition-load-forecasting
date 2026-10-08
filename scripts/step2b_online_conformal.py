"""Step 2b: online conformal calibration. On day d the operator knows realized loads through
day d-1, so nonconformity scores are updated daily. Compared: static split conformal (warm-start
calibration half only), rolling-window conformal (W days), ACI on top of rolling window,
and rolling temperature scaling. Applied to the Ours predictions saved in step 2."""
from common import *
import json
LOG = open(f"{OUT}/step2b_log.txt","w")
def log(s): print(s, flush=True); LOG.write(s+"\n"); LOG.flush()

def qhat(scores, level):
    n=len(scores); k=min(n,int(np.ceil((n+1)*level))); return np.sort(scores)[k-1]

def run(name):
    P = np.load(f"{OUT}/{name}_step2_preds.npz"); sc, mn = float(P["scale"]), float(P["mean"])
    y = P["tel"]*sc+mn; mu = P["ours_mu"]*sc+mn; sig = np.sqrt(P["ours_ale"]**2+P["ours_epi"]**2)*sc
    ycal = P["call"]*sc+mn; mucal = P["ours_mu_cal"]*sc+mn; sigcal = P["ours_tot_cal"]*sc
    s_cal = np.abs(ycal-mucal)/sigcal; s_te = np.abs(y-mu)/sig
    N=len(y); days=[np.arange(i,min(i+24,N)) for i in range(0,N,24)]
    log(f"\n=== {name}: {N} test points, {len(days)} days; warm-start calibration scores n={len(s_cal)}")
    z = norm.ppf((1+LEVELS)/2)
    methods = {"Raw (Gaussian)": None, "Static split conformal": ("static",None),
               "Rolling conformal W=30d": ("roll",30), "Rolling conformal W=90d": ("roll",90),
               "ACI (gamma=0.01) on W=30d": ("aci",30), "Rolling temperature W=30d": ("temp",30)}
    res={}; curves={}; roll_cov={}
    for mname, spec in methods.items():
        cover = np.zeros((N,len(LEVELS))); width90 = np.zeros(N)
        alpha_t = 1-LEVELS.copy(); gamma=0.01
        hist_s = list(s_cal)           # realized scores available to the operator
        for di, idx in enumerate(days):
            if spec is None: mult = z
            else:
                kind, W = spec
                if kind=="static": pool = np.array(s_cal)
                else:
                    pool = np.array(hist_s[-W*24:]) if len(hist_s)>=W*24 else np.array(hist_s)
                if kind=="temp":
                    tau = np.sqrt(np.mean(pool**2)); mult = z*tau
                elif kind=="aci":
                    mult = np.array([qhat(pool, min(max(1-a,0.01),0.999)) for a in alpha_t])
                else:
                    mult = np.array([qhat(pool, c) for c in LEVELS])
            for j,c in enumerate(LEVELS):
                cover[idx,j] = s_te[idx] <= mult[j]
            width90[idx] = 2*mult[-2]*sig[idx]
            if spec is not None and spec[0]=="aci":
                err = 1-cover[idx].mean(0)                     # daily miscoverage per level
                alpha_t = alpha_t + gamma*((1-LEVELS) - err)   # Gibbs & Candès update (batch of 24)
            hist_s += list(s_te[idx])                          # day d realized -> available for d+1
        obs = cover.mean(0); ece = np.mean(np.abs(LEVELS-obs)); picp90 = obs[-2]
        lo = mu - width90/2; hi = mu + width90/2; a=0.1
        is90 = np.mean((hi-lo)+2/a*(lo-y)*(y<lo)+2/a*(y-hi)*(y>hi))
        res[mname]=dict(ECE=ece, PICP90=picp90, IS90=is90, width90=np.mean(width90))
        curves[mname]=obs.tolist()
        # rolling 30-day 90% coverage trajectory
        c90 = cover[:,-2]; roll_cov[mname]=[float(np.mean(c90[max(0,i-30*24):i+24])) for i in range(0,N,24)]
        log(f"{mname:28s} ECE={ece:.4f} PICP90={picp90:.4f} IS90={is90:.5g} mean width90={np.mean(width90):.5g}")
    json.dump(dict(res=res, curves=curves, roll_cov=roll_cov), open(f"{OUT}/{name}_step2b.json","w"), indent=1)
    return res, curves, roll_cov

if __name__=="__main__":
    R={}
    for name in ["GEF","COVID"]: R[name]=run(name)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    sty={"Raw (Gaussian)":('#888','-'),"Static split conformal":('#2ca02c','-'),"Rolling conformal W=30d":('#1f77b4','-'),"ACI (gamma=0.01) on W=30d":('#d62728','--')}
    for ax,name,tag in zip(axes,["GEF","COVID"],["(a) GEFCom2014","(b) post-COVID"]):
        rc=R[name][2]
        for m,(c,ls) in sty.items(): ax.plot(np.arange(len(rc[m])), rc[m], color=c, ls=ls, lw=1.6, label=m)
        ax.axhline(0.9,color='k',lw=1,ls=':'); ax.set_ylim(0.6,1.0); ax.set_xlabel("Test day"); ax.set_ylabel("Trailing 30-day coverage of 90% interval")
        ax.set_title(tag,loc='left',fontweight='bold'); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/step2b_online_conformal.png",dpi=180,bbox_inches="tight")
    log("DONE")
