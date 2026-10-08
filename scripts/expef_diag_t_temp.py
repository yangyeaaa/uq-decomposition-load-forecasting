"""Experiment e: residual diagnostics of the Gaussian heteroscedastic model (K=10 ensemble) and a
Student-t likelihood variant (K=5). Experiment f: adding target-day temperature as a 7th variate."""
from common import *
import torch.nn as nn
from scipy.stats import kstest, skew, kurtosis, t as student_t
import json
LOG=open(f"{OUT}/expef_log.txt","a")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()
COLS={"GEF":dict(hour=0,load=1,temp=2),"COVID":dict(load=0,temp=1,hour=4)}

def members(name,d,X,K=10,prefix=None):
    mus,vs=[],[]
    for k in range(K):
        m=make_model(d,0.1); m.load_state_dict(torch.load(f"{OUT}/{prefix or name+'_ens'}{k}.pth",weights_only=True))
        r=predict(m,X,1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
    mus,vs=np.stack(mus),np.stack(vs); return mus.mean(0),np.sqrt(vs.mean(0)),np.sqrt(mus.var(0))

# ---------------- e1: residual diagnostics ----------------
for name in ([] if os.environ.get("SKIP_E1") else ["GEF","COVID"]):
    log(f"\n{'='*70}\n[e1] {name}: residual diagnostics, K=10 ensemble, test partition\n{'='*70}")
    d=prep_daystrat(name); c=COLS[name]; sc=d["sdl"].scale_[0]
    mu,ale,epi=members(name,d,d["te"]); y=d["tel"]; e=y-mu; tot=np.sqrt(ale**2+epi**2); z=e/tot
    X=d["te"]; hour=X[:,-1,c["hour"]]; zT=X[:,-1,c["temp"]]; Y=y*sc+d["sdl"].mean_[0]
    log("raw residual std (MW) by group -> evidence of conditional heteroscedasticity")
    def grp(lab,key,edges):
        parts=[]
        for i in range(len(edges)-1):
            msk=(key>=edges[i])&(key<edges[i+1]) if i<len(edges)-2 else (key>=edges[i])&(key<=edges[i+1])
            parts.append(f"{np.std(e[msk])*sc:6.0f}")
        log(f"  {lab:14s}"+" ".join(parts))
    log(f"  {'hour bins':14s}"+" ".join(f"{int(h):>6d}" for h in np.linspace(hour.min(),hour.max(),7)[:-1])); grp("residual std",hour,np.linspace(hour.min(),hour.max()+1e-9,7))
    log(f"  {'temp bins':14s}"+" ".join(f"{q:>6.1f}" for q in np.percentile(zT,[0,17,33,50,67,83])) ); grp("residual std",zT,np.percentile(zT,[0,17,33,50,67,83,100]))
    log(f"  {'load bins':14s}"+" ".join(f"{q:>6.0f}" for q in np.percentile(Y,[0,17,33,50,67,83]))); grp("residual std",Y,np.percentile(Y,[0,17,33,50,67,83,100]))
    log(f"  ratio of largest to smallest group std: hour {np.max([np.std(e[(hour>=a)&(hour<b)]) for a,b in zip(np.linspace(hour.min(),hour.max()+1e-9,7)[:-1],np.linspace(hour.min(),hour.max()+1e-9,7)[1:])])/np.min([np.std(e[(hour>=a)&(hour<b)]) for a,b in zip(np.linspace(hour.min(),hour.max()+1e-9,7)[:-1],np.linspace(hour.min(),hour.max()+1e-9,7)[1:])]):.2f}")
    # did the variance head capture it? std of standardized residual by group should be ~1 everywhere
    log("std of standardized residual e/sigma_total by temperature bin (should be ~1 if the variance head captures heteroscedasticity):")
    ed=np.percentile(zT,[0,17,33,50,67,83,100]); log("  "+" ".join(f"{np.std(z[(zT>=ed[i])&(zT<=ed[i+1])]):6.2f}" for i in range(6)))
    pit=norm.cdf(z); ks=kstest(pit,'uniform')
    log(f"standardized residuals: mean={z.mean():+.3f} std={z.std():.3f} skew={skew(z):+.3f} excess kurtosis={kurtosis(z):+.3f} | PIT KS stat={ks.statistic:.3f} (p={ks.pvalue:.1e})")
    hist=np.histogram(pit,bins=10,range=(0,1))[0]/len(pit); log("PIT histogram (10 bins, uniform=0.10): "+" ".join(f"{h:.3f}" for h in hist))
    q=np.percentile(z,[1,5,25,50,75,95,99]); log("empirical quantiles of z vs N(0,1): "+" ".join(f"{a:+.2f}/{b:+.2f}" for a,b in zip(q,norm.ppf([.01,.05,.25,.5,.75,.95,.99]))))

# ---------------- e2: Student-t likelihood ----------------
def fwd(base,x):
    mu,logv,_=base(x); return mu,logv

def train_t(d,seed,save,epochs=50,patience=12):
    MM.setup_seed(seed); base=make_model(d,0.1); base.set_schedule(MM.DiffusionSchedule(5,DEV)); raw_nu=torch.nn.Parameter(torch.tensor(2.0))
    if os.path.exists(save):
        ck=torch.load(save,weights_only=True); base.load_state_dict(ck["base"]); return base,float(2.0+torch.nn.functional.softplus(ck["raw_nu"]))
    params=list(base.parameters())+[raw_nu]; opt=torch.optim.AdamW(params,lr=5e-4,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=5,factor=0.5,min_lr=1e-6)
    tl,vl=loaders(d); best=1e9; pc=0
    def nll(mu,logv,y,nu):
        s2=torch.exp(logv); return (-torch.lgamma((nu+1)/2)+torch.lgamma(nu/2)+0.5*torch.log(nu*np.pi*s2)+(nu+1)/2*torch.log1p((y-mu)**2/(nu*s2))).mean()
    for ep in range(epochs):
        base.train()
        for x,y in tl:
            mu,logv=fwd(base,x); nu=2.0+torch.nn.functional.softplus(raw_nu); loss=nll(mu,logv,y.squeeze(),nu)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        base.eval(); vr=[]
        with torch.no_grad():
            for x,y in vl: mu,logv=fwd(base,x); vr.append(torch.sqrt(((mu-y.squeeze())**2).mean()).item())
        vr=float(np.mean(vr)); sch.step(vr)
        if vr<best: best=vr; pc=0; torch.save({"base":base.state_dict(),"raw_nu":raw_nu.detach().clone()},save)
        else:
            pc+=1
            if pc>=patience: break
    ck=torch.load(save,weights_only=True); base.load_state_dict(ck["base"]); return base,float(2.0+torch.nn.functional.softplus(ck["raw_nu"]))

def t_metrics(y,mu,s,nu,sdl):
    sc,mn=sdl.scale_[0],sdl.mean_[0]; Y=y*sc+mn; P=mu*sc+mn; S=s*sc; z=(Y-P)/S
    m=dict(MAE=np.mean(np.abs(Y-P)),RMSE=np.sqrt(np.mean((Y-P)**2)))
    m["NLL"]=np.mean(-student_t.logpdf(Y,nu,loc=P,scale=S))
    rng=np.random.RandomState(0); smp=student_t.rvs(nu,size=(400,len(Y)),random_state=rng)*S+P
    m["CRPS"]=np.mean(np.mean(np.abs(smp-Y),0)-0.5*np.mean(np.abs(smp[:200]-smp[200:]),0))
    qs=np.arange(0.05,0.96,0.05); m["Pinball"]=np.mean([np.mean(np.maximum(q*(Y-(P+student_t.ppf(q,nu)*S)),(q-1)*(Y-(P+student_t.ppf(q,nu)*S)))) for q in qs])
    obs=np.array([np.mean(np.abs(z)<=student_t.ppf((1+c)/2,nu)) for c in LEVELS]); m["ECE"]=np.mean(np.abs(LEVELS-obs)); m["PICP90"]=obs[-2]
    m["PIT_KS"]=kstest(student_t.cdf(z,nu),'uniform').statistic; return m

for name in ["GEF","COVID"]:
    log(f"\n{'='*70}\n[e2] {name}: Student-t heteroscedastic likelihood, K=5 ensemble (mixture approximated by pooled scale)\n{'='*70}")
    d=prep_daystrat(name); sdl=d["sdl"]; mus,ss,nus=[],[],[]
    for k in range(5):
        base,nu=train_t(d,10+k,f"{OUT}/{name}_t_ens{k}.pth"); base.eval()
        with torch.no_grad(): mu,logv=fwd(base,torch.tensor(d["te"],dtype=torch.float32))
        mus.append(mu.numpy()); ss.append(np.exp(0.5*logv.numpy())); nus.append(nu)
    mus,ss=np.stack(mus),np.stack(ss); nu=float(np.mean(nus)); mu=mus.mean(0); s_ale=np.sqrt((ss**2).mean(0)); s_epi=np.sqrt(mus.var(0))
    # scale of the t predictive: sqrt(scale^2 + epistemic var) (approximation of the mixture)
    mt=t_metrics(d["tel"],mu,np.sqrt(s_ale**2+s_epi**2),nu,sdl)
    mg_mu,mg_ale,mg_epi=members(name,d,d["te"],5); mg=full_metrics(d["tel"],mg_mu,np.sqrt(mg_ale**2+mg_epi**2),sdl); mg["PIT_KS"]=kstest(norm.cdf((d["tel"]-mg_mu)/np.sqrt(mg_ale**2+mg_epi**2)),'uniform').statistic
    keys=["MAE","RMSE","CRPS","NLL","Pinball","ECE","PICP90","PIT_KS"]
    log(f"{'':22s}"+"".join(f"{k:>10s}" for k in keys)); log(f"{'Gaussian K=5':22s}"+"".join(f"{mg[k]:>10.4g}" for k in keys)); log(f"{'Student-t K=5 (nu=%.1f)'%nu:22s}"+"".join(f"{mt[k]:>10.4g}" for k in keys))
    sc=sdl.scale_[0]; log(f"decomposition under t: sigma_ale(scale)={s_ale.mean()*sc:.1f}  sigma_epi={s_epi.mean()*sc:.1f}  R_epi={np.mean(s_epi**2/(s_epi**2+s_ale**2)):.3f}  | Gaussian: sigma_ale={mg_ale.mean()*sc:.1f} sigma_epi={mg_epi.mean()*sc:.1f} R_epi={np.mean(mg_epi**2/(mg_epi**2+mg_ale**2)):.3f}")
    log(f"(t-scale is not a std; std = scale*sqrt(nu/(nu-2)) = {s_ale.mean()*sc*np.sqrt(nu/(nu-2)):.1f} MW)")

# ---------------- f: target-day temperature ----------------
def prep_tdt(name, add, seed=0):
    """Same as prep_daystrat but each partition is augmented with the target-day (same-hour) temperature
    as a 7th variate (repeated over the 7 steps); last 24 samples of each partition are dropped for alignment."""
    r=load_raw(name); T=COLS[name]["temp"]
    def aug(X,y):
        Xa=X[:-24]; ya=y[:-24]
        if add: tdt=X[24:,-1,T][:,None,None]*np.ones((1,X.shape[1],1)); Xa=np.concatenate([Xa,tdt],2)
        return Xa,ya
    tr,trl=aug(*r["train"]); vd,vdl=aug(*r["val"]); te,tel=aug(*r["test"]); sl,nv=tr.shape[1],tr.shape[2]
    sdd=StandardScaler().fit(tr.reshape(-1,nv)); sdl=StandardScaler().fit(trl.reshape(-1,1))
    f=lambda a: sdd.transform(a.reshape(-1,nv)).reshape(-1,sl,nv); g=lambda a: sdl.transform(a.reshape(-1,1)).flatten()
    rng=np.random.RandomState(seed); ies,ical=[],[]
    for d0 in range(0,len(vd),24):
        idx=np.arange(d0,min(d0+24,len(vd))); rng.shuffle(idx); h=len(idx)//2; ies+=list(idx[:h]); ical+=list(idx[h:])
    ies,ical=np.sort(ies),np.sort(ical)
    return dict(tr=f(tr),trl=g(trl),es=f(vd[ies]),esl=g(vdl[ies]),cal=f(vd[ical]),call=g(vdl[ical]),te=f(te),tel=g(tel),sl=sl,nv=nv,sdl=sdl)

for name in ["GEF","COVID"]:
    log(f"\n{'='*70}\n[f] {name}: adding target-day temperature (K=5 ensembles, both trained on identical samples)\n{'='*70}")
    res={}
    for add in [False,True]:
        d=prep_tdt(name,add); sc=d["sdl"].scale_[0]; mus,vs=[],[]
        for k in range(5):
            m=make_model(d,0.1,seed=10+k); path=f"{OUT}/{name}_tdt{int(add)}_ens{k}.pth"
            if os.path.exists(path): m.load_state_dict(torch.load(path,weights_only=True))
            else: train(m,d,10+k,path)
            r=predict(m,d["te"],1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
        mus,vs=np.stack(mus),np.stack(vs); mu=mus.mean(0); ale=np.sqrt(vs.mean(0)); epi=np.sqrt(mus.var(0)); tot=np.sqrt(ale**2+epi**2)
        m=full_metrics(d["tel"],mu,tot,d["sdl"]); res[add]=dict(m=m,ale=ale.mean()*sc,epi=epi.mean()*sc,R=np.mean(epi**2/tot**2))
        log(f"{'with target-day temp' if add else 'without (baseline)':22s} MAE={m['MAE']:.1f} CRPS={m['CRPS']:.1f} NLL={m['NLL']:.3f} PICP90={m['PICP90']:.3f} | sigma_ale={res[add]['ale']:.1f} sigma_epi={res[add]['epi']:.1f} R_epi={res[add]['R']:.3f}")
    log(f"change: MAE {100*(res[True]['m']['MAE']/res[False]['m']['MAE']-1):+.1f}%  sigma_ale {100*(res[True]['ale']/res[False]['ale']-1):+.1f}%  sigma_epi {100*(res[True]['epi']/res[False]['epi']-1):+.1f}%")
log("DONE")
