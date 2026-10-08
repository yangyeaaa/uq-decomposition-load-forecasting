"""Experiment b: K sensitivity of the ensemble decomposition (real GEF and semi-synthetic)."""
from common import *
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
import json
LOG=open(f"{OUT}/expb_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()
KMAX=20; KS=[3,5,10,20]

def members(prefix, d, X, seeds):
    mus,vs,tt=[],[],0.0
    for k,sd in enumerate(seeds):
        m=make_model(d,0.1,seed=sd); path=f"{OUT}/{prefix}{k}.pth"
        if os.path.exists(path): m.load_state_dict(torch.load(path,weights_only=True))
        else: tt+=train(m,d,sd,path)
        r=predict(m,X,1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
    return np.stack(mus),np.stack(vs),tt

def decomp(mus,vs,idx): return mus[idx].mean(0), np.sqrt(vs[idx].mean(0)), np.sqrt(mus[idx].var(0))

# ---------------- real data ----------------
log("="*70+"\n[1] GEF real data: K sensitivity (members = seeds 10..29)\n"+"="*70)
d=prep_daystrat("GEF"); sc=d["sdl"].scale_[0]
t0=time.time(); mus,vs,tt=members("GEF_ens",d,d["te"],list(range(10,10+KMAX))); log(f"training time for the 15 new members: {tt:.0f}s")
log(f"{'K':>3s}{'sig_epi':>9s}{'sig_ale':>9s}{'R_epi':>7s}{'MAE':>8s}{'CRPS':>8s}{'NLL':>8s}{'PICP90':>8s}{'ECE':>8s}{'rho(epi) 2 disjoint sets':>26s}{'top10% overlap':>16s}")
rows=[]
for K in KS:
    mu,ale,epi=decomp(mus,vs,list(range(K))); tot=np.sqrt(ale**2+epi**2); m=full_metrics(d["tel"],mu,tot,d["sdl"])
    if 2*K<=KMAX:
        _,_,e1=decomp(mus,vs,list(range(K))); _,_,e2=decomp(mus,vs,list(range(K,2*K)))
        rho=spearmanr(e1,e2)[0]; ov=np.mean((e1>=np.quantile(e1,.9))[e2>=np.quantile(e2,.9)])
    else: rho,ov=np.nan,np.nan
    R=np.mean(epi**2/tot**2); rows.append(dict(K=K,epi=epi.mean()*sc,ale=ale.mean()*sc,R=R,**m,rho=rho,ov=ov))
    log(f"{K:3d}{epi.mean()*sc:9.1f}{ale.mean()*sc:9.1f}{R:7.3f}{m['MAE']:8.1f}{m['CRPS']:8.1f}{m['NLL']:8.3f}{m['PICP90']:8.3f}{m['ECE']:8.3f}{rho:26.3f}{ov:16.3f}")
# unbiased estimate of the K->inf epistemic: sample variance already uses 1/K (np.var); report ddof=1 version too
e20=np.sqrt(mus.var(0,ddof=1)); log(f"K=20 with ddof=1 (unbiased member variance): mean sig_epi={e20.mean()*sc:.1f} MW")
# stability of the per-input epistemic ranking vs the K=20 reference as K grows
ref=np.sqrt(mus.var(0)); log("rho(sigma_epi(K), sigma_epi(K=20)) : "+", ".join(f"K={K}: {spearmanr(decomp(mus,vs,list(range(K)))[2],ref)[0]:.3f}" for K in [3,5,10]))
# error correlation
err=np.abs(d["tel"]-mus.mean(0)); log("rho(sigma_epi(K), |e|): "+", ".join(f"K={K}: {spearmanr(decomp(mus,vs,list(range(K)))[2],err)[0]:.3f}" for K in KS))

# ---------------- semi-synthetic ----------------
log("\n"+"="*70+"\n[2] semi-synthetic (known sigma(x), thinned region): K sensitivity\n"+"="*70)
from step4_synthetic import S, dd, Xte, sparse, xnorm, knn   # reuses the same synthetic labels (seeded) and split
mus_s,vs_s,tt=members("syn_ens",dd,Xte,list(range(10,10+KMAX))); log(f"training time for the new synthetic members: {tt:.0f}s")
f,s=S["te"]["f"],S["te"]["s"]
log(f"{'K':>3s}{'ale rho':>9s}{'ale bias':>9s}{'epi dense':>10s}{'epi sparse':>11s}{'ratio':>7s}{'true err ratio':>15s}{'AUC sparse>dense':>17s}{'rho(epi,|mu-f|)':>16s}{'rho(epi,kNN)':>13s}{'dense rho(epi,|x|)':>19s}")
for K in KS:
    mu,ale,epi=decomp(mus_s,vs_s,list(range(K))); errm=np.abs(mu-f)
    auc=roc_auc_score(sparse,epi)
    log(f"{K:3d}{spearmanr(ale,s)[0]:9.3f}{np.mean(ale/s)-1:+9.3f}{epi[~sparse].mean()*sc:10.1f}{epi[sparse].mean()*sc:11.1f}{epi[sparse].mean()/epi[~sparse].mean():7.2f}{errm[sparse].mean()/errm[~sparse].mean():15.2f}{auc:17.3f}{spearmanr(epi,errm)[0]:16.3f}{spearmanr(epi,knn)[0]:13.3f}{spearmanr(epi[~sparse],xnorm[~sparse])[0]:19.3f}")
log(f"reference: dense-region rho(|mu-f|,|x|) = {spearmanr(np.abs(mus_s.mean(0)-f)[~sparse],xnorm[~sparse])[0]:.3f}; true error ratio uses the K=20 mean")
# does the epistemic scale converge to the true posterior-mean error? compare mean sig_epi with RMS(mu_K - f) in each region
mu20=mus_s.mean(0)
for reg,msk in [("dense",~sparse),("sparse",sparse)]:
    log(f"{reg}: RMS true mean error (K=20) = {np.sqrt(np.mean((mu20[msk]-f[msk])**2))*sc:.1f} MW ; mean sigma_epi K=5 {np.sqrt(mus_s[:5].var(0))[msk].mean()*sc:.1f}, K=20 {np.sqrt(mus_s.var(0))[msk].mean()*sc:.1f} MW")
json.dump(rows,open(f"{OUT}/expb.json","w"),indent=1,default=float)
log("DONE")
