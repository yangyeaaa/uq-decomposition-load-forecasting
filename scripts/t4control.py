"""T4 control: train on the COMPLETE synthetic set (no thinning) and compare per-input epistemic
estimates with the thinned models; T3 variant: shift all seven lag temperatures jointly."""
from common import *
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
LOG=open(f"{OUT}/t4control_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()
from step4_synthetic import S, dd, Xte, sparse, xnorm, rng   # thinned data dd; S has the untrimmed synthetic labels? (S[k] were thinned in place)
# rebuild the FULL synthetic partitions with the same labels: regenerate deterministically
d=prep_daystrat("GEF"); sc=d["sdl"].scale_[0]
base=make_model(d,0.1); base.load_state_dict(torch.load(f"{OUT}/GEF_ours_seed1.pth",weights_only=True))
f_true=lambda X: predict(base,X,1,False)["mu"]
def sigma_true(X): zT=X[:,-1,2]; return 0.12+0.25/(1+np.exp(-2*zT))
rng2=np.random.RandomState(123); full={}
for k in ["tr","es","cal","te"]:
    f=f_true(d[k]); s=sigma_true(d[k]); full[k]=(d[k], f+s*rng2.randn(len(f)), f, s)
assert np.allclose(full["te"][1], S["te"]["y"]), "synthetic test labels differ"   # same rng sequence as step4
dfull=dict(d); dfull.update(tr=full["tr"][0],trl=full["tr"][1],es=full["es"][0],esl=full["es"][1],cal=full["cal"][0],call=full["cal"][1],te=full["te"][0],tel=full["te"][1])
f_te,s_te=full["te"][2],full["te"][3]
cont=Xte[:,:,[1,2]].reshape(len(Xte),-1); ncont=np.linalg.norm(cont,axis=1); dense=~sparse

def ens(prefix,dset,K,seeds):
    mus,vs=[],[]
    for k,sd in zip(range(K),seeds):
        m=make_model(dset,0.1,seed=sd); p=f"{OUT}/{prefix}{k}.pth"
        if os.path.exists(p): m.load_state_dict(torch.load(p,weights_only=True))
        else: train(m,dset,sd,p)
        r=predict(m,Xte,1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
    mus,vs=np.stack(mus),np.stack(vs); return mus.mean(0),np.sqrt(vs.mean(0)),np.sqrt(mus.var(0,ddof=1))
def mcd(path,dset,seed):
    m=make_model(dset,0.1,seed=seed)
    if os.path.exists(path): m.load_state_dict(torch.load(path,weights_only=True))
    else: train(m,dset,seed,path)
    r=predict(m,Xte,40,True); return r["mu"],r["ale"],r["epi"]*np.sqrt(40/39)

log("[T4 control] full-data vs thinned training; Delta = epi_thin - epi_full per test input")
log(f"{'estimator':22s}{'epi full: dense/sparse':>24s}{'epi thin: dense/sparse':>24s}{'mean Delta dense':>17s}{'mean Delta sparse':>18s}{'Delta ratio':>12s}{'AUC(Delta)':>11s}{'rho(Delta,|x|cont) dense':>25s}")
res={}
# MC dropout p=0.1
mu_t,ale_t,epi_t=mcd(f"{OUT}/syn_p10.pth",dd,1); mu_f,ale_f,epi_f=mcd(f"{OUT}/synfull_p10.pth",dfull,1)
D=epi_t-epi_f; res["mcd"]=D
log(f"{'MC dropout p=0.1':22s}{epi_f[dense].mean()*sc:11.1f}/{epi_f[sparse].mean()*sc:<11.1f}{epi_t[dense].mean()*sc:11.1f}/{epi_t[sparse].mean()*sc:<11.1f}{D[dense].mean()*sc:17.1f}{D[sparse].mean()*sc:18.1f}{D[sparse].mean()/max(D[dense].mean(),1e-9):12.2f}{roc_auc_score(sparse,D):11.3f}{spearmanr(D[dense],ncont[dense])[0]:25.3f}")
# ensemble K=5 (thinned members seeds 10..14 exist as syn_ens0..4)
mu_t,ale_t,epi_t=ens("syn_ens",dd,5,range(10,15)); mu_f,ale_f,epi_f=ens("synfull_ens",dfull,5,range(10,15))
D=epi_t-epi_f; res["ens"]=D
log(f"{'Ensemble K=5':22s}{epi_f[dense].mean()*sc:11.1f}/{epi_f[sparse].mean()*sc:<11.1f}{epi_t[dense].mean()*sc:11.1f}/{epi_t[sparse].mean()*sc:<11.1f}{D[dense].mean()*sc:17.1f}{D[sparse].mean()*sc:18.1f}{D[sparse].mean()/max(D[dense].mean(),1e-9):12.2f}{roc_auc_score(sparse,D):11.3f}{spearmanr(D[dense],ncont[dense])[0]:25.3f}")
# true change in estimation error
err_t=np.abs(mu_t-f_te); err_f=np.abs(mu_f-f_te); dE=err_t-err_f
log(f"true change in |mu-f| (ensemble): dense {dE[dense].mean()*sc:+.1f} MW, sparse {dE[sparse].mean()*sc:+.1f} MW; AUC(dE)={roc_auc_score(sparse,dE):.3f}")
log(f"aleatoric recovery on full data (ensemble): rho={spearmanr(ale_f,s_te)[0]:.3f} bias={np.mean(ale_f/s_te)-1:+.3f}")
np.savez(f"{OUT}/t4control.npz",D_mcd=res["mcd"],D_ens=res["ens"],sparse=sparse)

log("\n[T3 variant] shift all seven lag temperatures jointly by delta (K=10 ensemble, real GEF)")
tmax=d["tr"][:,:,2].max()
def ens10(X):
    mus,vs=[],[]
    for k in range(10):
        m=make_model(d,0.1); m.load_state_dict(torch.load(f"{OUT}/GEF_ens{k}.pth",weights_only=True)); r=predict(m,X,1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
    mus,vs=np.stack(mus),np.stack(vs); return np.sqrt(vs.mean(0)),np.sqrt(mus.var(0,ddof=1))
log(f"{'delta':>6s}{'sig_epi':>9s}{'sig_ale':>9s}{'R_epi':>7s}{'frac beyond train max':>23s}   (last-lag-only shift in brackets)")
for delta in [0,0.5,1.0,1.5,2.0]:
    X=d["te"].copy(); X[:,:,2]+=delta; ale,epi=ens10(X)
    X1=d["te"].copy(); X1[:,-1,2]+=delta; ale1,epi1=ens10(X1)
    log(f"{delta:6.1f}{epi.mean()*sc:9.1f}{ale.mean()*sc:9.1f}{np.mean(epi**2/(epi**2+ale**2)):7.3f}{np.mean(X[:,:,2].max(1)>tmax):23.3f}   [{epi1.mean()*sc:.1f} / {ale1.mean()*sc:.1f} / {np.mean(epi1**2/(epi1**2+ale1**2)):.3f}]")
log("DONE")
