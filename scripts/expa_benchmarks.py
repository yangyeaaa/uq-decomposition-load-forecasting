"""Experiment a: statistical / ML benchmarks on the same inputs as the neural models
(7 same-hour daily lags x 6 variates, no target-day weather).
 1. Linear heteroscedastic regression (OLS location, log-linear scale, two-stage)
 2. GAMLSS-type additive model: spline terms in location and log-scale, alternating weighted fits
 3. Gradient boosting: LightGBM quantile regression (19 quantiles) and LightGBM location-scale
"""
from common import *
import json, warnings; warnings.filterwarnings("ignore")
from pygam import LinearGAM, s, f as fac, te
import lightgbm as lgb
LOG=open(f"{OUT}/expa_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()
COLS={"GEF":dict(hour=0,load=1,temp=2,month=3,day=4,wday=5),"COVID":dict(load=0,temp=1,month=2,day=3,hour=4,wday=5)}

def design(X, c):
    """Flattened window + target-day calendar (same hour; weekday advanced by one day)."""
    n=len(X); F=X.reshape(n,-1)
    hour=X[:,-1,c["hour"]]; wd=(X[:,-1,c["wday"]]+1)%7; month=X[:,-1,c["month"]]
    return np.c_[F,hour,wd,month]

def gaussian_rows(name, y, mu, sig, sdl, extra=None):
    m=full_metrics(y,mu,sig,sdl); return m

def fit_linear(Xtr,ytr,Xte):
    from sklearn.linear_model import Ridge
    A=Ridge(alpha=1.0).fit(Xtr,ytr); r=ytr-A.predict(Xtr)
    B=Ridge(alpha=1.0).fit(Xtr,np.log(r**2+1e-6))
    sig_tr=np.sqrt(np.exp(B.predict(Xtr)+1.2704)); tau=np.sqrt(np.mean((r/sig_tr)**2))
    return A.predict(Xte), np.sqrt(np.exp(B.predict(Xte)+1.2704))*tau

def gam_terms(nfeat, c, hour_col, wd_col, month_col):
    # spline on every continuous window feature, factor terms for target hour/weekday/month, hour x weekday interaction
    t=None
    for j in range(nfeat):
        term=s(j,n_splines=8,lam=10.0); t=term if t is None else t+term
    t=t+fac(hour_col)+fac(wd_col)   # no month factor: redundant given 7 lagged loads and harmful under shift
    return t

def fit_gamlss(Xtr,ytr,Xte,c,n_iter=3):
    """Normal-family GAMLSS approximation: alternate (i) weighted spline fit of the location with
    weights 1/sigma^2 and (ii) spline fit of log residual variance."""
    n_win=Xtr.shape[1]-3; hc,wc,mc=n_win,n_win+1,n_win+2
    # drop calendar columns from the spline block: hour/wday/month of lag days are near-constant within window; keep loads & temps only
    keep=[j for j in range(n_win) if j%6 in (c["load"],c["temp"])]
    Xs_tr=np.c_[Xtr[:,keep],Xtr[:,[hc,wc,mc]]]; Xs_te=np.c_[Xte[:,keep],Xte[:,[hc,wc,mc]]]
    k=len(keep); hc,wc,mc=k,k+1,k+2
    lo,hi=Xs_tr.min(0),Xs_tr.max(0); Xs_te=np.clip(Xs_te,lo,hi)
    terms=gam_terms(k,c,hc,wc,mc)
    w=np.ones(len(ytr)); logv=None
    for it in range(n_iter):
        gm=LinearGAM(terms).fit(Xs_tr,ytr,weights=w)
        r=ytr-gm.predict(Xs_tr)
        gv=LinearGAM(terms).fit(Xs_tr,np.log(r**2+1e-6))
        logv=gv.predict(Xs_tr)+1.2704; w=1.0/np.exp(logv)
    sig_tr=np.sqrt(np.exp(logv)); tau=np.sqrt(np.mean((r/sig_tr)**2))
    mu=gm.predict(Xs_te); sig=np.sqrt(np.exp(gv.predict(Xs_te)+1.2704))*tau; return mu,sig,gm.statistics_["edof"]

def fit_lgb_quantile(Xtr,ytr,Xte,qs,Xes=None,yes=None):
    Q=[]
    for q in qs:
        m=lgb.LGBMRegressor(objective="quantile",alpha=q,n_estimators=600,learning_rate=0.05,num_leaves=15,min_child_samples=50,verbose=-1)
        m.fit(Xtr,ytr,eval_set=[(Xes,yes)],callbacks=[lgb.early_stopping(30,verbose=False)]); Q.append(m.predict(Xte))
    return np.sort(np.stack(Q,1),1)

def fit_lgb_locscale(Xtr,ytr,Xte):
    A=lgb.LGBMRegressor(n_estimators=400,learning_rate=0.05,num_leaves=31,verbose=-1).fit(Xtr,ytr)
    # out-of-fold residuals for the scale model to avoid in-sample overfit of the mean
    from sklearn.model_selection import KFold
    r=np.zeros(len(ytr))
    for tr_i,va_i in KFold(5,shuffle=True,random_state=0).split(Xtr):
        r[va_i]=ytr[va_i]-lgb.LGBMRegressor(n_estimators=400,learning_rate=0.05,num_leaves=31,verbose=-1).fit(Xtr[tr_i],ytr[tr_i]).predict(Xtr[va_i])
    B=lgb.LGBMRegressor(n_estimators=100,learning_rate=0.05,num_leaves=7,min_child_samples=100,verbose=-1).fit(Xtr,np.log(r**2+1e-6))
    sig_tr=np.sqrt(np.exp(B.predict(Xtr)+1.2704)); tau=np.sqrt(np.mean((r/sig_tr)**2))
    return A.predict(Xte), np.sqrt(np.exp(B.predict(Xte)+1.2704))*tau

def quantile_metrics(Q, qs, y, sdl):
    sc,mn=sdl.scale_[0],sdl.mean_[0]; Y=y*sc+mn; Q=Q*sc+mn
    qs=np.asarray(qs); ix=lambda q: int(np.argmin(np.abs(qs-q))); med=Q[:,ix(0.5)]; e=Y-med
    m=dict(MAE=np.mean(np.abs(e)),RMSE=np.sqrt(np.mean(e**2)),MAPE=100*np.mean(np.abs(e/Y)))
    pin=np.array([np.mean(np.maximum(q*(Y-Q[:,i]),(q-1)*(Y-Q[:,i]))) for i,q in enumerate(qs)])
    m["CRPS"]=2*np.mean(pin); m["NLL"]=np.nan; m["Pinball"]=np.mean([pin[ix(q)] for q in np.arange(0.05,0.96,0.05)])
    lo,hi=Q[:,ix(0.05)],Q[:,ix(0.95)]; a=0.1; m["IS90"]=np.mean((hi-lo)+2/a*(lo-Y)*(Y<lo)+2/a*(Y-hi)*(Y>hi))
    obs=[]
    for c in LEVELS:
        il=ix((1-c)/2); ih=ix((1+c)/2); obs.append(np.mean((Y>=Q[:,il])&(Y<=Q[:,ih])))
    obs=np.array(obs); m["ECE"]=np.mean(np.abs(LEVELS-obs)); m["PICP90"]=obs[-2]; return m

def run(name):
    log(f"\n{'='*70}\n{name}\n{'='*70}")
    d=prep_daystrat(name); c=COLS[name]; sdl=d["sdl"]
    Xtr=design(d["tr"],c); Xte=design(d["te"],c); ytr=d["trl"]; yte=d["tel"]; Xcal=design(d["cal"],c); ycal=d["call"]
    rows={}; times={}
    def add_conf(label,fitfun):
        mu_c,sg_c=fitfun(Xcal)[:2]; return mu_c,sg_c
    t0=time.time(); mu,sg=fit_linear(Xtr,ytr,np.r_[Xte,Xcal]); times["Linear heteroscedastic"]=time.time()-t0
    nte=len(Xte); rows["Linear heteroscedastic"]=full_metrics(yte,mu[:nte],sg[:nte],sdl)
    cm=conformal_metrics(ycal,mu[nte:],sg[nte:],yte,mu[:nte],sg[:nte],sdl); rows["Linear heteroscedastic +conformal"]={**rows["Linear heteroscedastic"],"IS90":cm["IS90_conf"],"ECE":cm["ECE_conf"],"PICP90":cm["PICP90_conf"]}
    t0=time.time(); mu,sg,edof=fit_gamlss(Xtr,ytr,np.r_[Xte,Xcal],c); times["GAMLSS (additive, normal)"]=time.time()-t0
    rows["GAMLSS (additive, normal)"]=full_metrics(yte,mu[:nte],sg[:nte],sdl); log(f"GAMLSS effective dof (location) = {edof:.0f}")
    cm=conformal_metrics(ycal,mu[nte:],sg[nte:],yte,mu[:nte],sg[:nte],sdl); rows["GAMLSS +conformal"]={**rows["GAMLSS (additive, normal)"],"IS90":cm["IS90_conf"],"ECE":cm["ECE_conf"],"PICP90":cm["PICP90_conf"]}
    qs=np.round(np.arange(0.025,0.976,0.025),3)
    t0=time.time(); Q=fit_lgb_quantile(Xtr,ytr,Xte,qs,design(d["es"],c),d["esl"]); times["LightGBM quantile (39 q)"]=time.time()-t0
    rows["LightGBM quantile (39 q)"]=quantile_metrics(Q,qs,yte,sdl)
    t0=time.time(); mu,sg=fit_lgb_locscale(Xtr,ytr,np.r_[Xte,Xcal]); times["LightGBM location-scale"]=time.time()-t0
    rows["LightGBM location-scale"]=full_metrics(yte,mu[:nte],sg[:nte],sdl)
    cm=conformal_metrics(ycal,mu[nte:],sg[nte:],yte,mu[:nte],sg[:nte],sdl); rows["LightGBM location-scale +conformal"]={**rows["LightGBM location-scale"],"IS90":cm["IS90_conf"],"ECE":cm["ECE_conf"],"PICP90":cm["PICP90_conf"]}
    # reference rows from step 2 (neural, same split)
    S2=json.load(open(f"{OUT}/{name}_step2.json"))
    for k in ["Ours (raw)","Ours (+conformal)","Deep Ensemble (K=5)","Quantile Regression (median for MAE)"]: rows["[NN] "+k]=S2["rows"][k]
    keys=["MAE","RMSE","MAPE","CRPS","NLL","Pinball","IS90","ECE","PICP90"]
    log(f"{'Method':42s}"+"".join(f"{k:>10s}" for k in keys)+f"{'fit s':>8s}")
    for r,m in rows.items():
        t=times.get(r,S2["times"].get(r.replace("[NN] ",""),{}).get("train",np.nan))
        log(f"{r:42s}"+"".join(f"{m[k]:>10.4g}" if not (isinstance(m[k],float) and np.isnan(m[k])) else f"{'—':>10s}" for k in keys)+f"{t:8.1f}")
    json.dump(dict(rows={r:{k:float(v) for k,v in m.items()} for r,m in rows.items()},times=times),open(f"{OUT}/{name}_expa.json","w"),indent=1)

if __name__=="__main__":
    for name in ["GEF","COVID"]: run(name)
    log("DONE")
