"""Step 4: semi-synthetic reference. Real GEF covariates; y = f(x) + sigma(x)*eps with f = the
step-2 network's deterministic mean and sigma(x) a KNOWN function of last-lag temperature.
Training/validation are thinned (keep 10%) where z_T > 0.8; test is complete.
Compares MC-dropout (several p) and a K=5 ensemble as the epistemic source."""
from common import *
from scipy.stats import spearmanr, pearsonr
from sklearn.neighbors import NearestNeighbors
import json
LOG=open(f"{OUT}/step4_log.txt","w")
def log(s): print(s,flush=True); LOG.write(s+"\n"); LOG.flush()
T_COL=2; THIN_Z=0.8; KEEP=0.10; SEED=1

d=prep_daystrat("GEF"); sc=d["sdl"].scale_[0]
# true mean function from the step-2 model (deterministic)
base=make_model(d,0.1); base.load_state_dict(torch.load(f"{OUT}/GEF_ours_seed1.pth",weights_only=True))
f_true=lambda X: predict(base,X,n_mc=1,mc=False)["mu"]
def sigma_true(X): zT=X[:,-1,T_COL]; return 0.12+0.25/(1+np.exp(-2*zT))
rng=np.random.RandomState(123)
def synth(X):
    f=f_true(X); s=sigma_true(X); return f, s, f+s*rng.randn(len(f))
def thin(X,y,f,s):
    zT=X[:,-1,T_COL]; keep=(zT<=THIN_Z)|(rng.rand(len(X))<KEEP); return X[keep],y[keep],f[keep],s[keep],keep
S={}
for k in ["tr","es","cal","te"]:
    f,s,y=synth(d[k]); S[k]=dict(X=d[k],f=f,s=s,y=y)
for k in ["tr","es","cal"]:
    X,y,f,s,keep=thin(S[k]["X"],S[k]["y"],S[k]["f"],S[k]["s"]); log(f"{k}: {keep.sum()}/{len(keep)} kept after thinning z_T>{THIN_Z}"); S[k].update(X=X,y=y,f=f,s=s)
dd=dict(d); dd.update(tr=S["tr"]["X"],trl=S["tr"]["y"],es=S["es"]["X"],esl=S["es"]["y"],cal=S["cal"]["X"],call=S["cal"]["y"],te=S["te"]["X"],tel=S["te"]["y"])
Xte=S["te"]["X"]; zT=Xte[:,-1,T_COL]; sparse=zT>THIN_Z; log(f"test: {sparse.sum()} of {len(zT)} points in the sparse region")
# density proxy: distance to 5th nearest training input (flattened standardized window)
nn_=NearestNeighbors(n_neighbors=5).fit(S["tr"]["X"].reshape(len(S["tr"]["X"]),-1)); knn=nn_.kneighbors(Xte.reshape(len(Xte),-1))[0][:,-1]
xnorm=np.linalg.norm(Xte.reshape(len(Xte),-1),axis=1)

def evaluate(tag, mu, ale, epi):
    f,s=S["te"]["f"],S["te"]["s"]; err_mean=np.abs(mu-f)         # true error of the mean estimate
    r=dict(tag=tag,
        ale_corr=pearsonr(ale,s)[0], ale_rho=spearmanr(ale,s)[0], ale_bias=float(np.mean(ale/s)-1),
        ale_bias_dense=float(np.mean(ale[~sparse]/s[~sparse])-1), ale_bias_sparse=float(np.mean(ale[sparse]/s[sparse])-1),
        epi_mean=epi.mean(), epi_dense=epi[~sparse].mean(), epi_sparse=epi[sparse].mean(),
        epi_ratio=epi[sparse].mean()/epi[~sparse].mean(),
        err_dense=err_mean[~sparse].mean(), err_sparse=err_mean[sparse].mean(), err_ratio=err_mean[sparse].mean()/err_mean[~sparse].mean(),
        rho_epi_err=spearmanr(epi,err_mean)[0], rho_epi_knn=spearmanr(epi,knn)[0],
        rho_epi_xnorm_dense=spearmanr(epi[~sparse],xnorm[~sparse])[0], rho_err_xnorm_dense=spearmanr(err_mean[~sparse],xnorm[~sparse])[0],
        auc_sparse=float(np.mean(epi[sparse][:,None]>epi[~sparse][None,:1000])),  # P(epi_sparse > epi_dense) on a subsample
        mae=np.mean(np.abs(mu-S["te"]["y"])))
    log(f"{tag:22s} | ale: corr={r['ale_corr']:.3f} rho={r['ale_rho']:.3f} bias={r['ale_bias']:+.3f} (dense {r['ale_bias_dense']:+.3f}, sparse {r['ale_bias_sparse']:+.3f})"
        f" | epi: dense={r['epi_dense']*sc:.1f} sparse={r['epi_sparse']*sc:.1f} ratio={r['epi_ratio']:.2f} (true mean-error ratio {r['err_ratio']:.2f})"
        f" | rho(epi,|mu-f|)={r['rho_epi_err']:.3f} rho(epi,kNN dist)={r['rho_epi_knn']:.3f} AUC(sparse>dense)={r['auc_sparse']:.3f}"
        f" | dense region: rho(epi,|x|)={r['rho_epi_xnorm_dense']:.3f} vs rho(|mu-f|,|x|)={r['rho_err_xnorm_dense']:.3f}")
    return r

log("\n[1] MC dropout at several p (seed 1)")
R=[]; preds={}
for p in [0.05,0.10,0.30]:
    m=make_model(dd,p,seed=SEED); path=f"{OUT}/syn_p{int(p*100)}.pth"
    if not os.path.exists(path): train(m,dd,SEED,path)
    else: m.load_state_dict(torch.load(path,weights_only=True))
    r=predict(m,Xte); R.append(evaluate(f"MC-dropout p={p}",r["mu"],r["ale"],r["epi"])); preds[f"mcd{int(p*100)}"]=r
log("\n[2] deep ensemble K=5 (heteroscedastic members, deterministic)")
mus,vs=[],[]
for k in range(5):
    m=make_model(dd,0.1,seed=10+k); path=f"{OUT}/syn_ens{k}.pth"
    if not os.path.exists(path): train(m,dd,10+k,path)
    else: m.load_state_dict(torch.load(path,weights_only=True))
    r=predict(m,Xte,n_mc=1,mc=False); mus.append(r["mu"]); vs.append(r["ale"]**2)
mus,vs=np.stack(mus),np.stack(vs)
R.append(evaluate("Ensemble K=5",mus.mean(0),np.sqrt(vs.mean(0)),np.sqrt(mus.var(0))))
# hybrid: ensemble members each with MC dropout (posterior = seeds x masks)
mus2,vs2=[],[]
for k in range(5):
    m=make_model(dd,0.1,seed=10+k); m.load_state_dict(torch.load(f"{OUT}/syn_ens{k}.pth",weights_only=True))
    r=predict(m,Xte,n_mc=20); mus2.append(r["mu"]); vs2.append(r["ale"]**2)
# epistemic of the hybrid: var of member-averaged means + mean of within-member MC variances
mus2,vs2=np.stack(mus2),np.stack(vs2)
log("\n[3] ensemble + MC dropout (K=5 x 20 masks)")
# recompute within-member epistemic
within=[]
for k in range(5):
    m=make_model(dd,0.1,seed=10+k); m.load_state_dict(torch.load(f"{OUT}/syn_ens{k}.pth",weights_only=True)); within.append(predict(m,Xte,n_mc=20)["epi"]**2)
R.append(evaluate("Ensemble+MCD",mus2.mean(0),np.sqrt(vs2.mean(0)),np.sqrt(mus2.var(0)+np.mean(within,0))))

log("\n[4] epistemic by temperature bin (MW): does each source rise only where data were removed?")
bins=[-3,-1,0,0.8,1.5,3]; lab=["z_T<-1","-1..0","0..0.8","0.8..1.5 (sparse)",">1.5 (sparse)"]
log(f"{'':22s}"+"".join(f"{l:>18s}" for l in lab)+f"{'n':>8s}")
rows={"true |mu-f| (ens)":np.abs(mus.mean(0)-S['te']['f']),"MCD p=0.1":preds["mcd10"]["epi"],"Ensemble":np.sqrt(mus.var(0)),"true sigma(x)":S["te"]["s"],"ale MCD p=0.1":preds["mcd10"]["ale"],"ale Ensemble":np.sqrt(vs.mean(0))}
for name,v in rows.items():
    log(f"{name:22s}"+"".join(f"{v[(zT>=bins[i])&(zT<bins[i+1])].mean()*sc:18.1f}" for i in range(5))+f"{'':>8s}")
log(f"{'count':22s}"+"".join(f"{int(((zT>=bins[i])&(zT<bins[i+1])).sum()):18d}" for i in range(5)))
json.dump([{k:float(v) if not isinstance(v,str) else v for k,v in r.items()} for r in R],open(f"{OUT}/step4.json","w"),indent=1)
np.savez(f"{OUT}/step4_preds.npz",zT=zT,sparse=sparse,f=S["te"]["f"],s=S["te"]["s"],y=S["te"]["y"],knn=knn,
         mcd10_epi=preds["mcd10"]["epi"],mcd10_ale=preds["mcd10"]["ale"],mcd10_mu=preds["mcd10"]["mu"],
         ens_epi=np.sqrt(mus.var(0)),ens_ale=np.sqrt(vs.mean(0)),ens_mu=mus.mean(0))
log("DONE")
