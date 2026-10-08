"""Section 6.2 numbers: K=10 ensemble, Bessel-corrected epistemic, causal (lagged) rolling calibration,
paired 7-day moving-block bootstrap over target days for all incremental-value statistics."""
from common import *
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
B=1000; BLOCK=int(os.environ.get("BLOCK","7"))
LOG=open(f"{OUT}/sec62_block{BLOCK}_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()
def qhat(s,c): n=len(s); k=min(n,int(np.ceil((n+1)*c))); return np.sort(s)[k-1]
def mbb_days(nd,rng):
    """moving-block bootstrap of day indices, blocks of BLOCK consecutive days"""
    out=[]
    while len(out)<nd:
        s=rng.randint(0,nd-BLOCK+1); out+=list(range(s,s+BLOCK))
    return np.array(out[:nd])
def ci(stat_fn,nd,rng):
    vals=[stat_fn(mbb_days(nd,rng)) for _ in range(B)]; vals=np.array(vals); return np.percentile(vals,2.5),np.percentile(vals,97.5)

def run(name):
    d=prep_daystrat(name); sc,mn=d["sdl"].scale_[0],d["sdl"].mean_[0]; div=1000 if name=="COVID" else 1
    P=np.load(f"{OUT}/{name}_ensK10_preds.npz"); f=np.sqrt(10/9)
    mu,ale,epi=P["mu"],P["ale"],P["epi"]*f; muc,alec,epic=P["muc"],P["alec"],P["epic"]*f; y,yc=P["y"],P["yc"]
    tot=np.sqrt(ale**2+epi**2); totc=np.sqrt(alec**2+epic**2); R=epi**2/tot**2; err=np.abs(y-mu)
    N=len(y); nd=int(np.ceil(N/24)); day=np.arange(N)//24; rng=np.random.RandomState(0)
    hours_of=lambda days_idx: np.concatenate([np.where(day==dd)[0] for dd in days_idx])
    log(f"\n{'='*70}\n{name}: data-rich regime, K=10, Bessel, MW (post-COVID /1000)\n{'='*70}")
    # ---------- D1 ----------
    scores={"sigma_epi":epi,"R_epi":R,"sigma_total":tot,"sigma_ale":ale,"random":np.random.RandomState(1).rand(N)}
    top10=err>=np.quantile(err,.9); z95=norm.ppf(.95); failA=err>z95*ale
    def unfl(s,idx,x): e=err[idx]; ss=s[idx]; return e[ss<np.quantile(ss,1-x)].mean()*sc/div
    def capt(s,idx,x): e=err[idx]; ss=s[idx]; t=e>=np.quantile(e,.9); fl=ss>=np.quantile(ss,1-x); return (t&fl).sum()/t.sum()
    log("[D1] review triage. MAE of unflagged hours (MW) at x=5/10/20/30%; captured share of top-10% errors at x=10%; AUC(top10 err); AUC(ale-interval fail)")
    for lab,s in scores.items():
        allidx=np.arange(N)
        log(f"  {lab:12s} MAE|unfl: "+" ".join(f"{unfl(s,allidx,x):7.2f}" for x in [.05,.1,.2,.3])+f"   capt10={capt(s,allidx,.1):.3f}   AUC_top10={roc_auc_score(top10,s):.3f}   AUC_alefail={roc_auc_score(failA,s):.3f}")
    for x in [.1,.3]:
        lo,hi=ci(lambda di: unfl(epi,hours_of(di),x)-unfl(tot,hours_of(di),x),nd,rng)
        log(f"  paired diff MAE|unfl (epi - total) at x={x:.0%}: {unfl(epi,np.arange(N),x)-unfl(tot,np.arange(N),x):+.2f} MW  [95% CI {lo:+.2f}, {hi:+.2f}]")
    lo,hi=ci(lambda di: roc_auc_score(failA[hours_of(di)],epi[hours_of(di)])-roc_auc_score(failA[hours_of(di)],tot[hours_of(di)]),nd,rng)
    log(f"  paired diff AUC(ale-fail) (epi - total): {roc_auc_score(failA,epi)-roc_auc_score(failA,tot):+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]")
    log(f"  base rate of aleatoric-interval failure: {failA.mean():.3f}")
    # ---------- D2 ----------
    log("[D2] reserve: S = mu + q_tau * sqrt(ale^2 + c^2 epi^2), q per c on calibration half; cost per hour in MW")
    Yc=yc*sc+mn; Yt=y*sc+mn; Mc=muc*sc+mn; Mt=mu*sc+mn
    for ratio in [2,5,10,20]:
        tau=ratio/(ratio+1.0); cost=lambda Y,S: (ratio*np.maximum(Y-S,0)+np.maximum(S-Y,0))
        pol={}
        for c in [0,0.5,1,1.5,2,3,4]:
            sgc=np.sqrt(alec**2+(c*epic)**2); q=np.quantile((yc-muc)/sgc,tau); sgt=np.sqrt(ale**2+(c*epi)**2)
            pol[c]=dict(cal=cost(Yc,Mc+q*sgc*sc).mean(),test=cost(Yt,Mt+q*sgt*sc))
        cs=min(pol,key=lambda c:pol[c]["cal"]); fixed=cost(Yt,Mt+np.quantile(Yc-Mc,tau))
        dcost=pol[cs]["test"]-pol[1]["test"]; gain=100*dcost.mean()/pol[1]["test"].mean()
        lo,hi=ci(lambda di: 100*dcost[hours_of(di)].mean()/pol[1]["test"][hours_of(di)].mean(),nd,rng)
        log(f"  {ratio:2d}:1  c*={cs:<3}  cost total-only={pol[1]['test'].mean()/div:8.2f}  ale-only={pol[0]['test'].mean()/div:8.2f}  reweighted={pol[cs]['test'].mean()/div:8.2f}  fixed={fixed.mean()/div:8.2f}   relative change vs total = {-gain:+.2f}% [95% CI {-hi:+.2f}, {-lo:+.2f}]")
    # ---------- D3 ----------
    log("[D3] warning of miscoverage: online rolling W=90 (lagged) 90% interval; daily scores vs daily miscoverage")
    s_cal=np.abs(yc-muc)/totc; s_te=err/tot; days=[np.arange(i,min(i+24,N)) for i in range(0,N,24)]
    hist=list(s_cal); pending=None; miss=np.zeros(N); halfw=np.zeros(N)
    for idx in days:
        pool=np.array(hist[-90*24:]); q=qhat(pool,0.90); miss[idx]=s_te[idx]>q; halfw[idx]=q*tot[idx]
        if pending is not None: hist+=list(pending)
        pending=s_te[idx]
    dM=np.array([miss[i].mean() for i in days]); dR=np.array([R[i].mean() for i in days]); dE=np.array([epi[i].mean() for i in days]); dH=np.array([halfw[i].mean() for i in days])
    def top20miss(score,di): sc_=score[di]; m=dM[di]; hi_=sc_>=np.quantile(sc_,.8); return m[hi_].mean()-m[~hi_].mean()
    for lab,sco in [("daily R_epi",dR),("daily sigma_epi",dE),("calibrated half-width",dH)]:
        rho=spearmanr(sco,dM)[0]; hi_=sco>=np.quantile(sco,.8)
        log(f"  {lab:22s} rho(score, miscoverage)={rho:+.3f}   miscoverage top-20% days={dM[hi_].mean():.3f} vs rest={dM[~hi_].mean():.3f}")
    rngd=np.random.RandomState(2); ndd=len(days)
    lo,hi=ci(lambda di: spearmanr(dR[di],dM[di])[0]-spearmanr(dH[di],dM[di])[0],ndd,rngd)
    log(f"  paired diff rho (R_epi - half-width): {spearmanr(dR,dM)[0]-spearmanr(dH,dM)[0]:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]")
    lo,hi=ci(lambda di: top20miss(dR,di)-top20miss(dH,di),ndd,rngd)
    log(f"  paired diff top-20% miscoverage enrichment (R_epi - half-width): {top20miss(dR,np.arange(ndd))-top20miss(dH,np.arange(ndd)):+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]")
    # ---------- interval contribution ----------
    picp_a=np.mean(err<=z95*ale); picp_t=np.mean(err<=z95*tot); wred=1-ale.mean()/tot.mean()
    lo,hi=ci(lambda di: np.mean(err[hours_of(di)]<=z95*tot[hours_of(di)])-np.mean(err[hours_of(di)]<=z95*ale[hours_of(di)]),nd,rng)
    log(f"[interval] raw Gaussian 90%: PICP ale-only={picp_a:.3f} total={picp_t:.3f}  dPICP={picp_t-picp_a:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]   width reduction without epi={100*wred:.1f}%   R_epi mean={R.mean():.3f}")
for n in ["GEF","COVID"]: run(n)
log("DONE")
