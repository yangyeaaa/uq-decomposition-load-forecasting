"""Step 6 (operational value of the epistemic component), ensemble K=5 as the posterior.
(1) review triage: risk-coverage curves and AUC for detecting periods where the aleatoric-only
    interval fails; (2) daily R_epi vs daily miscoverage; (3) epistemic-reweighted reserve vs
    total-variance reserve; (4) post-COVID: daily R_epi around the lockdown onset."""
from common import *
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
import json
LOG=open(f"{OUT}/step6_log.txt","w")
def log(s): print(s,flush=True); LOG.write(s+"\n"); LOG.flush()

def ens_predict(name, d, X):
    mus,vs=[],[]
    for k in range(5):
        m=make_model(d,0.1); m.load_state_dict(torch.load(f"{OUT}/{name}_ens{k}.pth",weights_only=True))
        r=predict(m,X,n_mc=1,mc=False); mus.append(r["mu"]); vs.append(r["ale"]**2)
    mus,vs=np.stack(mus),np.stack(vs); return mus.mean(0), np.sqrt(vs.mean(0)), np.sqrt(mus.var(0))

def risk_coverage(score, err, fracs):
    order=np.argsort(-score); n=len(err); out=[]
    for f in fracs:
        k=int(n*f); keep=np.ones(n,bool); keep[order[:k]]=False; out.append(err[keep].mean())
    return np.array(out)

def reserve_cost(y, S, ratio):
    cu,co=ratio,1.0; return np.mean(cu*np.maximum(y-S,0)+co*np.maximum(S-y,0))

def run(name):
    log(f"\n{'='*70}\n{name}\n{'='*70}")
    d=prep_daystrat(name); sc,mn=d["sdl"].scale_[0],d["sdl"].mean_[0]
    mu_c,ale_c,epi_c=ens_predict(name,d,d["cal"]); mu_t,ale_t,epi_t=ens_predict(name,d,d["te"])
    y_t=d["tel"]; y_c=d["call"]; tot_t=np.sqrt(ale_t**2+epi_t**2); tot_c=np.sqrt(ale_c**2+epi_c**2)
    err=np.abs(y_t-mu_t); R_t=epi_t**2/tot_t**2
    # MC-dropout (step-2 model) for contrast
    P=np.load(f"{OUT}/{name}_step2_preds.npz"); mcd_epi=P["ours_epi"]; mcd_mu=P["ours_mu"]; mcd_ale=P["ours_ale"]

    log("\n[1] review triage. Target A: absolute error. Target B: aleatoric-only interval failure (|e|/sigma_ale > z90).")
    z90=norm.ppf(0.95); failA=err/ale_t>z90
    fr=[0.0,0.05,0.10,0.20,0.30]
    scores={"sigma_epi (ens)":epi_t,"R_epi (ens)":R_t,"sigma_total (ens)":tot_t,"sigma_ale (ens)":ale_t,"sigma_epi (MC-dropout)":mcd_epi,"random":np.random.RandomState(0).rand(len(err))}
    log(f"{'score':26s}"+"".join(f"{'MAE|flag '+str(int(f*100))+'%':>14s}" for f in fr)+f"{'AUC(|e| top10%)':>17s}{'AUC(ale-int fail)':>19s}{'fail rate in top10%':>21s}")
    top10=err>=np.quantile(err,0.9)
    for k,s in scores.items():
        rc=risk_coverage(s,err,fr)*sc; flag=s>=np.quantile(s,0.9)
        log(f"{k:26s}"+"".join(f"{v:14.1f}" for v in rc)+f"{roc_auc_score(top10,s):17.3f}{roc_auc_score(failA,s):19.3f}{failA[flag].mean():21.3f}")
    log(f"(base rate of aleatoric-interval failure: {failA.mean():.3f})")
    log(f"rho(sigma_epi, |e|)={spearmanr(epi_t,err)[0]:.3f}  rho(sigma_ale, |e|)={spearmanr(ale_t,err)[0]:.3f}  rho(sigma_epi, |e|/sigma_ale)={spearmanr(epi_t,err/ale_t)[0]:.3f}  rho(MCD epi, |e|/sigma_ale)={spearmanr(mcd_epi,err/mcd_ale)[0]:.3f}")

    log("\n[2] daily R_epi vs daily miscoverage of the 90% interval (static conformal from the calibration half)")
    q90=conformal_q(y_c,mu_c,tot_c,0.90); miss=(np.abs(y_t-mu_t)>q90*tot_t)
    N=len(y_t); days=[np.arange(i,min(i+24,N)) for i in range(0,N,24)]
    dR=np.array([R_t[i].mean() for i in days]); dE=np.array([epi_t[i].mean() for i in days]); dM=np.array([miss[i].mean() for i in days]); dA=np.array([ale_t[i].mean() for i in days])
    log(f"rho(daily mean R_epi, daily miscoverage)={spearmanr(dR,dM)[0]:.3f}   rho(daily sigma_epi, miscov)={spearmanr(dE,dM)[0]:.3f}   rho(daily sigma_ale, miscov)={spearmanr(dA,dM)[0]:.3f}")
    hi=dR>=np.quantile(dR,0.8); log(f"miscoverage on the 20% highest-R_epi days = {dM[hi].mean():.3f} vs other days = {dM[~hi].mean():.3f} (nominal 0.10)")

    log("\n[3] reserve scheduling: S = mu + q(tau*) * sqrt(sigma_ale^2 + (c*sigma_epi)^2); c tuned on the calibration half")
    log(f"{'ratio':>6s}{'c*':>6s}{'total-only (c=1)':>18s}{'aleatoric-only (c=0)':>21s}{'epi-reweighted (c*)':>21s}{'fixed reserve':>15s}{'gain vs total':>14s}")
    C=[0,0.5,1,1.5,2,3,4]; gains={}
    for ratio in [2,5,10,20]:
        tau=ratio/(ratio+1.0); Yc=y_c*sc+mn; Yt=y_t*sc+mn; Mc=mu_c*sc+mn; Mt=mu_t*sc+mn
        def S_of(c,mu,ale,epi,q): return mu+q*np.sqrt(ale**2+(c*epi)**2)*sc
        costs_c={}
        for c in C:
            sg=np.sqrt(ale_c**2+(c*epi_c)**2); q=np.quantile((y_c-mu_c)/sg,tau)     # conformal-style quantile of the normalized residual on cal
            costs_c[c]=(reserve_cost(Yc,S_of(c,Mc,ale_c,epi_c,q),ratio),q)
        cstar=min(costs_c,key=lambda c:costs_c[c][0])
        def test_cost(c): return reserve_cost(Yt,S_of(c,Mt,ale_t,epi_t,costs_c[c][1]),ratio)
        fixed=reserve_cost(Yt,Mt+np.quantile(Yc-Mc,tau),ratio)
        t1,t0,ts=test_cost(1),test_cost(0),test_cost(cstar); gains[ratio]=100*(t1-ts)/t1
        log(f"{ratio:>4d}:1{cstar:6.1f}{t1:18.1f}{t0:21.1f}{ts:21.1f}{fixed:15.1f}{gains[ratio]:13.2f}%")

    if name=="COVID":
        log("\n[4] post-COVID: daily mean R_epi and sigma_epi over the validation period (in time order) and the test period")
        r=load_raw("COVID"); vd,vdl=r["val"]; nv=d["nv"]; sl=d["sl"]
        sdd=StandardScaler().fit(r["train"][0].reshape(-1,nv)); Xv=sdd.transform(vd.reshape(-1,nv)).reshape(-1,sl,nv)
        mu_v,ale_v,epi_v=ens_predict(name,d,Xv); Rv=epi_v**2/(epi_v**2+ale_v**2)
        Nv=len(Rv); dv=[np.arange(i,min(i+24,Nv)) for i in range(0,Nv,24)]
        # val starts 2019-12-25; lockdown onset ~2020-03-15 = day 81 of val; test starts 2020-03-25
        segs={"val 2019-12-25..2020-02-28 (pre)":dv[:66],"val 2020-02-29..03-14 (pre, late)":dv[66:81],"val 2020-03-15..03-30 (onset)":dv[81:],
              "test 2020-03-25..04-30 (lockdown)":days[:37],"test 05..06":days[37:98],"test 07..08":days[98:160],"test 09..11-05":days[160:]}
        for k,ix in segs.items():
            src=(Rv,epi_v,ale_v) if k.startswith("val") else (R_t,epi_t,ale_t)
            idx=np.concatenate(ix); log(f"{k:38s} mean R_epi={src[0][idx].mean():.3f}  sigma_epi={src[1][idx].mean()*sc:8.0f}  sigma_ale={src[2][idx].mean()*sc:8.0f}")
        np.savez(f"{OUT}/COVID_step6_timeseries.npz",R_val=Rv,epi_val=epi_v,ale_val=ale_v,R_test=R_t,epi_test=epi_t,ale_test=ale_t)
    json.dump(dict(gains=gains),open(f"{OUT}/{name}_step6.json","w"))

if __name__=="__main__":
    for name in ["GEF","COVID"]: run(name)
    log("DONE")
