"""Recompute the ensemble-based main results with K=10 (and K=5 for reference):
Table-4 rows (raw / +static conformal / +online conformal), and the operational tests of step 6."""
from common import *
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import json
LOG=open(f"{OUT}/k10_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()

def members(name,d,X,K):
    mus,vs=[],[]
    for k in range(K):
        m=make_model(d,0.1,seed=10+k); path=f"{OUT}/{name}_ens{k}.pth"
        if os.path.exists(path): m.load_state_dict(torch.load(path,weights_only=True))
        else: train(m,d,10+k,path)
        r=predict(m,X,1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
    mus,vs=np.stack(mus),np.stack(vs); return mus.mean(0),np.sqrt(vs.mean(0)),np.sqrt(mus.var(0))

def qhat(s,c): n=len(s); k=min(n,int(np.ceil((n+1)*c))); return np.sort(s)[k-1]

def online_conformal(y,mu,sig,s_cal,W=90):
    s_te=np.abs(y-mu)/sig; N=len(y); hist=list(s_cal); cov=np.zeros((N,len(LEVELS))); w90=np.zeros(N)
    for i in range(0,N,24):
        idx=np.arange(i,min(i+24,N)); pool=np.array(hist[-W*24:]); mult=np.array([qhat(pool,c) for c in LEVELS])
        for j in range(len(LEVELS)): cov[idx,j]=s_te[idx]<=mult[j]
        w90[idx]=2*mult[-2]*sig[idx]; hist+=list(s_te[idx])
    obs=cov.mean(0); lo,hi=mu-w90/2,mu+w90/2; a=0.1
    return dict(ECE=np.mean(np.abs(LEVELS-obs)),PICP90=obs[-2],IS90=np.mean((hi-lo)+2/a*(lo-y)*(y<lo)+2/a*(y-hi)*(y>hi)))

def run(name):
    d=prep_daystrat(name); sdl=d["sdl"]; sc,mn=sdl.scale_[0],sdl.mean_[0]; y=d["tel"]; yc=d["call"]
    log(f"\n{'='*70}\n{name}\n{'='*70}")
    keys=["MAE","RMSE","CRPS","NLL","Pinball","IS90","ECE","PICP90"]
    log(f"{'Ensemble':34s}"+"".join(f"{k:>10s}" for k in keys)+f"{'R_epi':>8s}{'sig_epi':>9s}")
    out={}
    for K in [5,10]:
        mu,ale,epi=members(name,d,d["te"],K); muc,alec,epic=members(name,d,d["cal"],K)
        tot=np.sqrt(ale**2+epi**2); totc=np.sqrt(alec**2+epic**2); R=np.mean(epi**2/tot**2)
        m=full_metrics(y,mu,tot,sdl); log(f"{'K=%d raw'%K:34s}"+"".join(f"{m[k]:>10.4g}" for k in keys)+f"{R:8.3f}{epi.mean()*sc:9.1f}")
        cm=conformal_metrics(yc,muc,totc,y,mu,tot,sdl); log(f"{'K=%d +static conformal'%K:34s}"+"".join(f"{(m[k] if k not in ('IS90','ECE','PICP90') else cm[k+'_conf']):>10.4g}" for k in keys))
        Y=y*sc+mn; P=mu*sc+mn; S=tot*sc; oc=online_conformal(Y,P,S,np.abs(yc-muc)/totc,W=90)
        log(f"{'K=%d +online conformal W=90d'%K:34s}"+"".join(f"{(m[k] if k not in ('IS90','ECE','PICP90') else oc[k]):>10.4g}" for k in keys))
        # aleatoric-only interval coverage (role of the epistemic term in coverage)
        log(f"   PICP90 of the aleatoric-only Gaussian interval: {np.mean(np.abs(y-mu)<=norm.ppf(0.95)*ale):.3f} vs total {m['PICP90']:.3f}")
        out[K]=dict(mu=mu,ale=ale,epi=epi,muc=muc,alec=alec,epic=epic)
        np.savez(f"{OUT}/{name}_ensK{K}_preds.npz",mu=mu,ale=ale,epi=epi,muc=muc,alec=alec,epic=epic,y=y,yc=yc,scale=sc,mean=mn)
    # operational tests with K=10 (same definitions as step 6)
    mu,ale,epi=out[10]["mu"],out[10]["ale"],out[10]["epi"]; muc,alec,epic=out[10]["muc"],out[10]["alec"],out[10]["epic"]
    tot=np.sqrt(ale**2+epi**2); err=np.abs(y-mu); top10=err>=np.quantile(err,.9); failA=err/ale>norm.ppf(.95)
    log("\n[operational, K=10] review triage: MAE of the unflagged 90% (MW) / AUC for top-10% errors / AUC for aleatoric-interval failure")
    for lab,s in [("sigma_epi",epi),("R_epi",epi**2/tot**2),("sigma_total",tot),("sigma_ale",ale),("random",np.random.RandomState(0).rand(len(err)))]:
        log(f"   {lab:12s} MAE|flag10%={err[s<np.quantile(s,.9)].mean()*sc:8.1f}   AUC(top10% err)={roc_auc_score(top10,s):.3f}   AUC(ale-int fail)={roc_auc_score(failA,s):.3f}")
    log(f"   rho(sigma_epi,|e|)={spearmanr(epi,err)[0]:.3f}  rho(sigma_ale,|e|)={spearmanr(ale,err)[0]:.3f}  rho(sigma_epi,|e|/sigma_ale)={spearmanr(epi,err/ale)[0]:.3f}")
    log("[operational, K=10] reserve: S = mu + q * sqrt(ale^2 + (c*epi)^2), c tuned on the calibration half")
    for ratio in [2,5,10,20]:
        tau=ratio/(ratio+1); Yc=yc*sc+mn; Yt=y*sc+mn; Mc=muc*sc+mn; Mt=mu*sc+mn
        cost=lambda Y,S: np.mean(ratio*np.maximum(Y-S,0)+np.maximum(S-Y,0)); res={}
        for c in [0,0.5,1,1.5,2,3,4]:
            sg=np.sqrt(alec**2+(c*epic)**2); q=np.quantile((yc-muc)/sg,tau); res[c]=(cost(Yc,Mc+q*sg*sc),cost(Yt,Mt+q*np.sqrt(ale**2+(c*epi)**2)*sc))
        cs=min(res,key=lambda c:res[c][0]); log(f"   {ratio:2d}:1  total-only={res[1][1]:.1f}  ale-only={res[0][1]:.1f}  epi-reweighted(c*={cs})={res[cs][1]:.1f}  gain vs total={100*(res[1][1]-res[cs][1])/res[1][1]:+.2f}%")
    if name=="COVID":
        N=len(y); days=[np.arange(i,min(i+24,N)) for i in range(0,N,24)]; R=epi**2/tot**2
        segs={"test 03-25..04-30 (lockdown)":days[:37],"05..06":days[37:98],"07..08":days[98:160],"09..11-05":days[160:]}
        log("[K=10] post-COVID segments: "+"; ".join(f"{k}: R_epi={R[np.concatenate(v)].mean():.3f}" for k,v in segs.items()))

if __name__=="__main__":
    for name in ["GEF","COVID"]: run(name)
    log("DONE")
