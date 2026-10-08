"""Section 6.3 numbers: data-scarce regime, K=5 ensemble per M, Bessel-corrected, causal rolling calibration,
paired 7-day moving-block bootstrap. Reuses the step-7 checkpoints and split."""
from common import *
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from step7_scarce import make_split
B=1000; BLOCK=7
LOG=open(f"{OUT}/sec63_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()
def qhat(s,c): n=len(s); k=min(n,int(np.ceil((n+1)*c))); return np.sort(s)[k-1]
def mbb(nd,rng):
    out=[]
    while len(out)<nd:
        s=rng.randint(0,max(1,nd-BLOCK+1)); out+=list(range(s,min(s+BLOCK,nd)))
    return np.array(out[:nd])
def ci(fn,nd,rng): v=np.array([fn(mbb(nd,rng)) for _ in range(B)]); return np.percentile(v,2.5),np.percentile(v,97.5)
def ens(name,M,d,X):
    mus,vs=[],[]
    for k in range(5):
        m=make_model(d,0.1); m.load_state_dict(torch.load(f"{OUT}/{name}_M{M}_ens{k}.pth",weights_only=True)); r=predict(m,X,1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
    mus,vs=np.stack(mus),np.stack(vs); return mus.mean(0),np.sqrt(vs.mean(0)),np.sqrt(mus.var(0,ddof=1))
z95=norm.ppf(.95)
for name in ["GEF","COVID"]:
    div=1000 if name=="COVID" else 1
    log(f"\n{'='*70}\n{name}: data-scarce regime (first M days, K=5, Bessel), MW\n{'='*70}")
    log(f"{'M':>4s}{'MAE':>8s}{'R_epi':>7s}{'sig_epi':>8s}{'sig_ale':>8s}{'PICP ale':>9s}{'PICP tot':>9s}{'dPICP [CI]':>24s}{'width red':>10s}{'D1 dMAE10 epi-tot [CI]':>28s}{'D1 AUC epi/tot':>16s}{'D2 10:1 gain [CI]':>24s}{'D3 rho R/half':>15s}")
    for M in [30,60,90,180,None]:
        d=make_split(name,M); sc,mn=d["sdl"].scale_[0],d["sdl"].mean_[0]
        mu,ale,epi=ens(name,M,d,d["te"]); muc,alec,epic=ens(name,M,d,d["cal"]); y,yc=d["tel"],d["call"]
        tot=np.sqrt(ale**2+epi**2); totc=np.sqrt(alec**2+epic**2); err=np.abs(y-mu); R=epi**2/tot**2
        N=len(y); nd=int(np.ceil(N/24)); day=np.arange(N)//24; rng=np.random.RandomState(0); H=lambda di: np.concatenate([np.where(day==dd)[0] for dd in di])
        mae=err.mean()*sc/div
        picp_a=np.mean(err<=z95*ale); picp_t=np.mean(err<=z95*tot); wred=1-ale.mean()/tot.mean()
        lo,hi=ci(lambda di: np.mean(err[H(di)]<=z95*tot[H(di)])-np.mean(err[H(di)]<=z95*ale[H(di)]),nd,rng)
        def unfl(s,idx,x): e=err[idx]; ss=s[idx]; return e[ss<np.quantile(ss,1-x)].mean()*sc/div
        d1=unfl(epi,np.arange(N),.1)-unfl(tot,np.arange(N),.1); l1,h1=ci(lambda di: unfl(epi,H(di),.1)-unfl(tot,H(di),.1),nd,rng)
        top10=err>=np.quantile(err,.9); auc_e=roc_auc_score(top10,epi); auc_t=roc_auc_score(top10,tot)
        # D2 at 10:1
        ratio=10; tau=ratio/(ratio+1.0); Yc=yc*sc+mn; Yt=y*sc+mn; Mc=muc*sc+mn; Mt=mu*sc+mn; cost=lambda Y,S: ratio*np.maximum(Y-S,0)+np.maximum(S-Y,0)
        pol={}
        for c in [0,0.5,1,1.5,2,3,4]:
            sgc=np.sqrt(alec**2+(c*epic)**2); q=np.quantile((yc-muc)/sgc,tau); pol[c]=(cost(Yc,Mc+q*sgc*sc).mean(),cost(Yt,Mt+q*np.sqrt(ale**2+(c*epi)**2)*sc))
        cs=min(pol,key=lambda c:pol[c][0]); dc=pol[cs][1]-pol[1][1]; gain=-100*dc.mean()/pol[1][1].mean()
        l2,h2=ci(lambda di: -100*dc[H(di)].mean()/pol[1][1][H(di)].mean(),nd,rng)
        # D3 rolling W=90 lagged
        s_cal=np.abs(yc-muc)/totc; s_te=err/tot; days=[np.arange(i,min(i+24,N)) for i in range(0,N,24)]
        hist=list(s_cal); pending=None; miss=np.zeros(N); halfw=np.zeros(N)
        for idx in days:
            pool=np.array(hist[-90*24:]); q=qhat(pool,0.90); miss[idx]=s_te[idx]>q; halfw[idx]=q*tot[idx]
            if pending is not None: hist+=list(pending)
            pending=s_te[idx]
        dM=np.array([miss[i].mean() for i in days]); dR=np.array([R[i].mean() for i in days]); dH=np.array([halfw[i].mean() for i in days])
        rR=spearmanr(dR,dM)[0]; rH=spearmanr(dH,dM)[0]
        log(f"{M or 359:4d}{mae:8.1f}{R.mean():7.3f}{epi.mean()*sc/div:8.1f}{ale.mean()*sc/div:8.1f}{picp_a:9.3f}{picp_t:9.3f}   {picp_t-picp_a:+.3f} [{lo:+.3f},{hi:+.3f}]{100*wred:9.1f}%   {d1:+6.2f} [{l1:+.2f},{h1:+.2f}]      {auc_e:.2f}/{auc_t:.2f}    c*={cs:<3} {gain:+.2f}% [{l2:+.2f},{h2:+.2f}]   {rR:+.2f}/{rH:+.2f}")
log("DONE")
