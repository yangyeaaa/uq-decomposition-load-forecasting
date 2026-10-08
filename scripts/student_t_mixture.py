"""Appendix E: recompute the Student-t rows of Table `tab:student` from the K=5 mixture
(1/K) sum_i t_{nu_i}(y; mu_i, s_i) instead of the single-t pooled-scale approximation.
Gaussian K=5 row recomputed with the same code path (Bessel-corrected epistemic variance)."""
from common import *
from scipy.stats import kstest, t as student_t, norm
import importlib.util, sys
spec=importlib.util.spec_from_file_location("ef","expef_diag_t_temp.py")
# reuse train_t/fwd without executing the module body: copy the two functions
src=open("expef_diag_t_temp.py").read()
exec(src[src.index("def fwd"):src.index("def t_metrics")], globals())
LOG=open(f"{OUT}/student_t_mixture_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()
qs19=np.arange(0.05,0.96,0.05)

def mix_cdf(z,MU,S,NU):            # z: (n,) or (n,G); MU,S: (K,n); NU: (K,)
    F=0
    for i in range(MU.shape[0]): F=F+student_t.cdf((z-MU[i][...,None] if z.ndim==2 else z-MU[i])/(S[i][...,None] if z.ndim==2 else S[i]),NU[i])
    return F/MU.shape[0]
def mix_quantile(p,MU,S,NU,iters=60):
    lo=(MU-12*S).min(0); hi=(MU+12*S).max(0)
    for _ in range(iters):
        mid=0.5*(lo+hi); F=mix_cdf(mid,MU,S,NU); lo=np.where(F<p,mid,lo); hi=np.where(F<p,hi,mid)
    return 0.5*(lo+hi)
def mix_crps(y,MU,S,NU,G=1200,chunk=400):
    out=np.zeros(len(y))
    for a in range(0,len(y),chunk):
        sl=slice(a,a+chunk); mu=MU[:,sl]; s=S[:,sl]; yy=y[sl]
        lo=np.minimum((mu-30*s).min(0),yy-1); hi=np.maximum((mu+30*s).max(0),yy+1)
        z=lo[:,None]+(hi-lo)[:,None]*np.linspace(0,1,G)[None,:]          # (n,G)
        F=mix_cdf(z,mu,s,NU); H=(z>=yy[:,None]).astype(float)
        out[sl]=np.trapezoid((F-H)**2,z,axis=1)
    return out
def mix_metrics(y,MU,S,NU,sdl):
    sc,mn=sdl.scale_[0],sdl.mean_[0]; K=MU.shape[0]
    med=mix_quantile(0.5,MU,S,NU); mubar=MU.mean(0)
    m={}
    m["MAE"]=np.mean(np.abs(y-mubar))*sc; m["RMSE"]=np.sqrt(np.mean((y-mubar)**2))*sc
    m["MAE_med"]=np.mean(np.abs(y-med))*sc
    m["CRPS"]=np.mean(mix_crps(y,MU,S,NU))*sc
    pdf=np.mean([student_t.pdf((y-MU[i])/S[i],NU[i])/S[i] for i in range(K)],0); m["NLL"]=np.mean(-np.log(pdf))+np.log(sc)
    Q={q:mix_quantile(q,MU,S,NU) for q in np.unique(np.concatenate([qs19,(1-LEVELS)/2,(1+LEVELS)/2]).round(6))}
    m["Pinball"]=np.mean([np.mean(np.maximum(q*(y-Q[round(q,6)]),(q-1)*(y-Q[round(q,6)]))) for q in qs19])*sc
    obs=np.array([np.mean((y>=Q[round((1-c)/2,6)])&(y<=Q[round((1+c)/2,6)])) for c in LEVELS])
    m["ECE"]=np.mean(np.abs(LEVELS-obs)); m["PICP90"]=obs[-2]
    lo,hi=Q[round(0.05,6)],Q[round(0.95,6)]; a=0.1; m["IS90"]=np.mean((hi-lo)+2/a*(lo-y)*(y<lo)+2/a*(y-hi)*(y>hi))*sc
    pit=mix_cdf(y,MU,S,NU); m["PIT_KS"]=kstest(pit,'uniform').statistic
    return m

for name in ["GEF","COVID"]:
    d=prep_daystrat(name); sdl=d["sdl"]; sc=sdl.scale_[0]; y=d["tel"]; Xte=torch.tensor(d["te"],dtype=torch.float32)
    log(f"\n{'='*70}\n{name}: Gaussian vs Student-t heteroscedastic likelihood, K=5 (seeds 10-14), test partition\n{'='*70}")
    # Gaussian members (same checkpoints as the main ensemble, members 0-4)
    MUg,Sg=[],[]
    for k in range(5):
        m=make_model(d,0.1); m.load_state_dict(torch.load(f"{OUT}/{name}_ens{k}.pth",weights_only=True)); r=predict(m,d["te"],1,False); MUg.append(r["mu"]); Sg.append(r["ale"])
    MUg,Sg=np.stack(MUg),np.stack(Sg); mug=MUg.mean(0); ale_g=np.sqrt((Sg**2).mean(0)); epi_g=np.sqrt(MUg.var(0,ddof=1)); tot_g=np.sqrt(ale_g**2+epi_g**2)
    mg=full_metrics(y,mug,tot_g,sdl); mg["PIT_KS"]=kstest(norm.cdf((y-mug)/tot_g),'uniform').statistic; Rg=np.mean(epi_g**2/tot_g**2)
    # Gaussian mixture (exact, for reference: moment-matched Gaussian is the paper's convention)
    mgm=mix_metrics(y,MUg,Sg,np.full(5,1e9),sdl)
    # Student-t members
    MUt,St,NU=[],[],[]
    for k in range(5):
        base,nu=train_t(d,10+k,f"{OUT}/{name}_t_ens{k}.pth"); base.eval()
        with torch.no_grad(): mu,logv=fwd(base,Xte)
        MUt.append(mu.numpy()); St.append(np.exp(0.5*logv.numpy())); NU.append(nu); log(f"  t member {k}: nu={nu:.2f}")
    MUt,St,NU=np.stack(MUt),np.stack(St),np.array(NU)
    mt=mix_metrics(y,MUt,St,NU,sdl)
    ale_t2=np.mean(St**2*(NU/(NU-2))[:,None],0); epi_t2=MUt.var(0,ddof=1); Rt=np.mean(epi_t2/(epi_t2+ale_t2))
    # old approximation for comparison
    nub=NU.mean(); s_pool=np.sqrt((St**2).mean(0)+MUt.var(0)); i0=src.index("def t_metrics"); exec(src[i0:src.index("\nfor name in",i0)], globals()); ns2={"t_metrics":t_metrics}
    mold=ns2["t_metrics"](y,MUt.mean(0),s_pool,nub,sdl)
    keys=["MAE","RMSE","CRPS","NLL","Pinball","IS90","ECE","PICP90","PIT_KS"]
    log(f"{'':40s}"+"".join(f"{k:>9s}" for k in keys)+f"{'R_epi':>8s}")
    log(f"{'Gaussian K=5, moment-matched (paper)':40s}"+"".join(f"{mg[k]:>9.4g}" for k in keys)+f"{Rg:8.3f}")
    log(f"{'Gaussian K=5, exact mixture':40s}"+"".join(f"{mgm.get(k,np.nan):>9.4g}" for k in keys)+f"{Rg:8.3f}")
    log(f"{'Student-t K=5, single-t approx (old)':40s}"+"".join(f"{mold.get(k,np.nan):>9.4g}" for k in keys)+f"{'':8s}")
    log(f"{'Student-t K=5, exact mixture (new)':40s}"+"".join(f"{mt[k]:>9.4g}" for k in keys)+f"{Rt:8.3f}")
    log(f"  mean learned nu = {nub:.2f} (members: {', '.join(f'{v:.2f}' for v in NU)});  MAE of mixture median = {mt['MAE_med']:.1f} MW")
    log(f"  decomposition under t (exact): sigma_ale={np.sqrt(ale_t2.mean())*sc:.1f} MW  sigma_epi={np.sqrt(epi_t2.mean())*sc:.1f} MW  R_epi={Rt:.3f} | Gaussian: sigma_ale={ale_g.mean()*sc:.1f} sigma_epi={epi_g.mean()*sc:.1f} R_epi={Rg:.3f}")
log("DONE")
