from common import *
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
LOG=open(f"{OUT}/t4control2_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()
from step4_synthetic import S, dd, Xte, sparse
d=prep_daystrat("GEF"); sc=d["sdl"].scale_[0]
base=make_model(d,0.1); base.load_state_dict(torch.load(f"{OUT}/GEF_ours_seed1.pth",weights_only=True))
f_true=lambda X: predict(base,X,1,False)["mu"]
def sigma_true(X): zT=X[:,-1,2]; return 0.12+0.25/(1+np.exp(-2*zT))
rng2=np.random.RandomState(123); full={}
for k in ["tr","es","cal","te"]:
    f=f_true(d[k]); s=sigma_true(d[k]); full[k]=(d[k], f+s*rng2.randn(len(f)), f, s)
dfull=dict(d); dfull.update(tr=full["tr"][0],trl=full["tr"][1],es=full["es"][0],esl=full["es"][1],cal=full["cal"][0],call=full["cal"][1],te=full["te"][0],tel=full["te"][1])
f_te=full["te"][2]; dense=~sparse
def member(path,dset,seed,p=0.1):
    m=make_model(dset,p,seed=seed)
    if os.path.exists(path): m.load_state_dict(torch.load(path,weights_only=True))
    else: train(m,dset,seed,path)
    return m
# ensemble members: thinned syn_ens{k} (seed 10+k), full synfull_ens{k} (seed 10+k)
mus_t,mus_f=[],[]
for k in range(10):
    mt=member(f"{OUT}/syn_ens{k}.pth",dd,10+k); mf=member(f"{OUT}/synfull_ens{k}.pth",dfull,10+k)
    mus_t.append(predict(mt,Xte,1,False)["mu"]); mus_f.append(predict(mf,Xte,1,False)["mu"]); log(f"  member {k} ready")
mus_t,mus_f=np.stack(mus_t),np.stack(mus_f)
def rep(idx):
    et=np.sqrt(mus_t[idx].var(0,ddof=1)); ef=np.sqrt(mus_f[idx].var(0,ddof=1)); D=et-ef
    return D[dense].mean()*sc, D[sparse].mean()*sc, (D[sparse].mean()-D[dense].mean())*sc, roc_auc_score(sparse,D)
log("\n[paired controlled comparison, ensemble] Delta = epi_thin - epi_full (MW); members paired by seed")
for lab,idx in [("K=5, members 0-4",list(range(5))),("K=5, members 5-9",list(range(5,10))),("K=10, members 0-9",list(range(10)))]:
    a,b,c,auc=rep(idx); log(f"  {lab:18s}: Delta unthinned={a:+.1f}  thinned={b:+.1f}  contrast={c:+.1f}  AUC={auc:.3f}")
# estimation-error change for K=10
et=np.abs(mus_t.mean(0)-f_te); ef=np.abs(mus_f.mean(0)-f_te); dE=et-ef
log(f"  Delta e (K=10): unthinned={dE[dense].mean()*sc:+.1f} thinned={dE[sparse].mean()*sc:+.1f} contrast={(dE[sparse].mean()-dE[dense].mean())*sc:+.1f} AUC={roc_auc_score(sparse,dE):.3f}")
log("\n[paired controlled comparison, MC dropout, M=40]")
for p in [0.05,0.10,0.30]:
    mt=member(f"{OUT}/syn_p{int(p*100)}.pth",dd,1,p); mf=member(f"{OUT}/synfull_p{int(p*100)}.pth",dfull,1,p)
    et=predict(mt,Xte,40,True)["epi"]*np.sqrt(40/39); ef=predict(mf,Xte,40,True)["epi"]*np.sqrt(40/39); D=et-ef
    log(f"  p={p:.2f}: Delta unthinned={D[dense].mean()*sc:+.1f}  thinned={D[sparse].mean()*sc:+.1f}  contrast={(D[sparse].mean()-D[dense].mean())*sc:+.1f}  AUC={roc_auc_score(sparse,D):.3f}")
log("DONE")
