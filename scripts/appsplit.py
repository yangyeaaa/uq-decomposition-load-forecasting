"""Appendix: robustness of the main results to the early-stopping / calibration split.
Main protocol: 12 h / 12 h within each target day. Alternative: alternating complete days.
K=10 ensembles retrained under the alternative split; same seeds."""
from common import *
import json
LOG=open(f"{OUT}/appsplit_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()

def prep_altday(name):
    r=load_raw(name); tr,trl=r["train"]; vd,vdl=r["val"]; te,tel=r["test"]; sl,nv=tr.shape[1],tr.shape[2]
    sdd=StandardScaler().fit(tr.reshape(-1,nv)); sdl=StandardScaler().fit(trl.reshape(-1,1))
    f=lambda a: sdd.transform(a.reshape(-1,nv)).reshape(-1,sl,nv); g=lambda a: sdl.transform(a.reshape(-1,1)).flatten()
    day=np.arange(len(vd))//24; ies=np.where(day%2==0)[0]; ical=np.where(day%2==1)[0]
    return dict(tr=f(tr),trl=g(trl),es=f(vd[ies]),esl=g(vdl[ies]),cal=f(vd[ical]),call=g(vdl[ical]),te=f(te),tel=g(tel),sl=sl,nv=nv,sdl=sdl,n_es=len(ies),n_cal=len(ical))

def members(prefix,d,X,K):
    mus,vs=[],[]
    for k in range(K):
        m=make_model(d,0.1,seed=10+k); path=f"{OUT}/{prefix}{k}.pth"
        if os.path.exists(path): m.load_state_dict(torch.load(path,weights_only=True))
        else: train(m,d,10+k,path)
        r=predict(m,X,1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
    mus,vs=np.stack(mus),np.stack(vs); return mus.mean(0),np.sqrt(vs.mean(0)),np.sqrt(mus.var(0))

def qhat(s,c): n=len(s); k=min(n,int(np.ceil((n+1)*c))); return np.sort(s)[k-1]
def online(y,mu,sig,s_cal,W=90):
    s_te=np.abs(y-mu)/sig; N=len(y); hist=list(s_cal); cov=np.zeros((N,len(LEVELS))); w=np.zeros(N)
    for i in range(0,N,24):
        idx=np.arange(i,min(i+24,N)); pool=np.array(hist[-W*24:]); mult=np.array([qhat(pool,c) for c in LEVELS])
        for j in range(len(LEVELS)): cov[idx,j]=s_te[idx]<=mult[j]
        w[idx]=2*mult[-2]*sig[idx]; hist+=list(s_te[idx])
    obs=cov.mean(0); lo,hi=mu-w/2,mu+w/2; a=0.1
    return dict(ECE=np.mean(np.abs(LEVELS-obs)),PICP90=obs[-2],IS90=np.mean((hi-lo)+2/a*(lo-y)*(y<lo)+2/a*(y-hi)*(y>hi)))

keys=["MAE","RMSE","CRPS","NLL","Pinball","IS90","ECE","PICP90"]
for name in ["GEF","COVID"]:
    log(f"\n{'='*70}\n{name}: K=10 ensemble, alternating-day split (early-stop days even, calibration days odd)\n{'='*70}")
    d=prep_altday(name); sdl=d["sdl"]; sc,mn=sdl.scale_[0],sdl.mean_[0]; y=d["tel"]; yc=d["call"]
    log(f"early-stop n={d['n_es']}  calibration n={d['n_cal']}")
    mu,ale,epi=members(f"{name}_alt_ens",d,d["te"],10); muc,alec,epic=members(f"{name}_alt_ens",d,d["cal"],10)
    tot=np.sqrt(ale**2+epi**2); totc=np.sqrt(alec**2+epic**2); R=np.mean(epi**2/tot**2)
    m=full_metrics(y,mu,tot,sdl); cm=conformal_metrics(yc,muc,totc,y,mu,tot,sdl)
    oc=online(y*sc+mn,mu*sc+mn,tot*sc,np.abs(yc-muc)/totc,90)
    log(f"{'':30s}"+"".join(f"{k:>10s}" for k in keys)+f"{'R_epi':>8s}{'sig_epi':>9s}{'sig_ale':>9s}")
    log(f"{'raw':30s}"+"".join(f"{m[k]:>10.4g}" for k in keys)+f"{R:8.3f}{epi.mean()*sc:9.1f}{ale.mean()*sc:9.1f}")
    log(f"{'+static conformal':30s}"+"".join(f"{(m[k] if k not in ('IS90','ECE','PICP90') else cm[k+'_conf']):>10.4g}" for k in keys))
    log(f"{'+online conformal W=90d':30s}"+"".join(f"{(m[k] if k not in ('IS90','ECE','PICP90') else oc[k]):>10.4g}" for k in keys))
    # reference: main protocol (12h/12h) numbers from the k10 log
    ref=[l for l in open(f"{OUT}/k10_log.txt").read().split(name)[1].split("=====")[0].split("\n") if l.startswith("K=10")]
    log("reference (main 12h/12h split):"); [log("   "+l) for l in ref]
log("DONE")
