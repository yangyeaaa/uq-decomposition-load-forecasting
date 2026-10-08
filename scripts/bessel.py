"""Bessel-corrected epistemic variance (ddof=1) for every ensemble result used in the paper.
Inference only, from saved checkpoints / prediction arrays under out/ (main 12h/12h protocol)."""
from common import *
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
LOG=open(f"{OUT}/bessel_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()

def members_pred(paths,d,X):
    mus,vs=[],[]
    for p in paths:
        m=make_model(d,0.1); m.load_state_dict(torch.load(p,weights_only=True)); r=predict(m,X,1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
    return np.stack(mus),np.stack(vs)
def decomp(mus,vs,idx):
    mu=mus[idx].mean(0); ale=np.sqrt(vs[idx].mean(0)); epi=np.sqrt(mus[idx].var(0,ddof=1)); return mu,ale,epi
def R(epi,ale): return np.mean(epi**2/(epi**2+ale**2))

# ---- 1. Table 4 rows (K=5, K=10) and PICP ale-only, both datasets ----
for name in ["GEF","COVID"]:
    d=prep_daystrat(name); sc=d["sdl"].scale_[0]; y=d["tel"]
    mus,vs=members_pred([f"{OUT}/{name}_ens{k}.pth" for k in range(10)],d,d["te"])
    log(f"\n[1] {name}: Table-4 ensemble rows with ddof=1")
    for K in [5,10]:
        mu,ale,epi=decomp(mus,vs,list(range(K))); tot=np.sqrt(ale**2+epi**2); m=full_metrics(y,mu,tot,d["sdl"])
        log(f"  K={K}: "+fmt(m)+f"  R_epi={R(epi,ale):.3f} sig_epi={epi.mean()*sc:.1f} sig_ale={ale.mean()*sc:.1f}  PICP90 ale-only={np.mean(np.abs(y-mu)<=norm.ppf(.95)*ale):.3f}")
    np.savez(f"{OUT}/{name}_bessel_K10.npz",mu=decomp(mus,vs,list(range(10)))[0],ale=decomp(mus,vs,list(range(10)))[1],epi=decomp(mus,vs,list(range(10)))[2])

# ---- 2. K sensitivity (GEF, 20 members) ----
d=prep_daystrat("GEF"); sc=d["sdl"].scale_[0]; y=d["tel"]
mus,vs=members_pred([f"{OUT}/GEF_ens{k}.pth" for k in range(20)],d,d["te"])
log("\n[2] GEF K sensitivity, ddof=1")
log(f"{'K':>3s}{'sig_epi':>9s}{'sig_ale':>9s}{'R_epi':>7s}{'MAE':>8s}{'CRPS':>8s}{'PICP90':>8s}{'rho disjoint':>13s}{'top10 overlap':>15s}")
for K in [3,5,10,20]:
    mu,ale,epi=decomp(mus,vs,list(range(K))); tot=np.sqrt(ale**2+epi**2); m=full_metrics(y,mu,tot,d["sdl"])
    if 2*K<=20:
        e1=decomp(mus,vs,list(range(K)))[2]; e2=decomp(mus,vs,list(range(K,2*K)))[2]; rho=spearmanr(e1,e2)[0]; ov=np.mean((e1>=np.quantile(e1,.9))[e2>=np.quantile(e2,.9)])
    else: rho=ov=np.nan
    log(f"{K:3d}{epi.mean()*sc:9.1f}{ale.mean()*sc:9.1f}{R(epi,ale):7.3f}{m['MAE']:8.1f}{m['CRPS']:8.1f}{m['PICP90']:8.3f}{rho:13.3f}{ov:15.3f}")

# ---- 3. Reference test (synthetic), ddof=1 ----
from step4_synthetic import S, dd, Xte, sparse, xnorm   # cached checkpoints; re-generates the same synthetic labels
mus_s,vs_s=members_pred([f"{OUT}/syn_ens{k}.pth" for k in range(20)],dd,Xte); f,s=S["te"]["f"],S["te"]["s"]
log("\n[3] reference test, ensemble ddof=1")
log(f"{'K':>3s}{'ale rho':>9s}{'ale bias':>9s}{'epi dense':>10s}{'epi sparse':>11s}{'ratio':>7s}{'AUC':>6s}{'rho(epi,|mu-f|)':>16s}{'dense rho(epi,|x|)':>19s}{'RMS err dense':>14s}{'RMS err sparse':>15s}")
for K in [3,5,10,20]:
    mu,ale,epi=decomp(mus_s,vs_s,list(range(K))); errm=np.abs(mu-f)
    log(f"{K:3d}{spearmanr(ale,s)[0]:9.3f}{np.mean(ale/s)-1:+9.3f}{epi[~sparse].mean()*sc:10.1f}{epi[sparse].mean()*sc:11.1f}{epi[sparse].mean()/epi[~sparse].mean():7.2f}{roc_auc_score(sparse,epi):6.3f}{spearmanr(epi,errm)[0]:16.3f}{spearmanr(epi[~sparse],xnorm[~sparse])[0]:19.3f}{np.sqrt(np.mean(errm[~sparse]**2))*sc:14.1f}{np.sqrt(np.mean(errm[sparse]**2))*sc:15.1f}")
mcd=np.load(f"{OUT}/step4_preds.npz"); e=mcd["mcd10_epi"]*np.sqrt(40/39)
log(f"MC dropout p=0.1 (M=40, ddof=1 factor 1.013): AUC={roc_auc_score(sparse,e):.3f} ratio={e[sparse].mean()/e[~sparse].mean():.2f} dense rho(epi,|x|)={spearmanr(e[~sparse],xnorm[~sparse])[0]:.3f}")

# ---- 4. Reducibility replicates (K=5), ddof=1 ----
log("\n[4] reducibility, K=5, 3 replicates, ddof=1")
res={}
for r in range(3):
    perm=np.random.RandomState(1000+r).permutation(len(d["trl"]))
    for fr in [0.10,0.25,0.50,0.75,1.0]:
        paths=[f"{OUT}/red_r{r}_f{int(fr*100)}_m{k}.pth" for k in range(5)]
        if not all(os.path.exists(p) for p in paths): continue
        mus_r,vs_r=members_pred(paths,d,d["te"]); mu,ale,epi=decomp(mus_r,vs_r,list(range(5))); tot=np.sqrt(ale**2+epi**2)
        res.setdefault(fr,[]).append(dict(epi=epi.mean()*sc,ale=ale.mean()*sc,R=R(epi,ale),PICPa=np.mean(np.abs(y-mu)<=norm.ppf(.95)*ale),PICPt=np.mean(np.abs(y-mu)<=norm.ppf(.95)*tot)))
for fr,L in res.items():
    A={k:np.array([x[k] for x in L]) for k in L[0]}
    log(f"  frac {fr:.2f}: sig_epi={A['epi'].mean():.1f}±{A['epi'].std(ddof=1):.1f}  sig_ale={A['ale'].mean():.1f}±{A['ale'].std(ddof=1):.1f}  R_epi={A['R'].mean():.3f}±{A['R'].std(ddof=1):.3f}  PICP ale-only/total={A['PICPa'].mean():.3f}/{A['PICPt'].mean():.3f}")

# ---- 5. Scarce regime (K=5 per M), ddof=1 ----
log("\n[5] data-scarce regime, K=5, ddof=1 (sig_epi x1.118; R_epi recomputed from saved per-input arrays)")
for name in ["GEF","COVID"]:
    for M in [30,60,90,180,None]:
        p=f"{OUT}/{name}_M{M}_preds.npz"
        if not os.path.exists(p): continue
        P=np.load(p); epi=P["epi"]*np.sqrt(5/4); ale=P["ale"]; y=P["y"]; mu=P["mu"]; tot=np.sqrt(ale**2+epi**2)
        log(f"  {name} M={M or 359}: R_epi={R(epi,ale):.3f} sig_epi={epi.mean()*float(P['scale']):.1f} PICP90 ale-only={np.mean(np.abs(y-mu)<=norm.ppf(.95)*ale):.3f} total={np.mean(np.abs(y-mu)<=norm.ppf(.95)*tot):.3f}")
log("DONE")
