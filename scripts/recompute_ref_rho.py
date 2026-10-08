"""Recompute Table `tab:reference` exactly: all seven estimators on the thinned semi-synthetic data,
with the continuous-covariate norm (14 dims: 7 lagged loads + 7 lagged temperatures) as defined in the
paper, Bessel-corrected ensemble epistemic variance, and the Table `tab:tempbins` ensemble row."""
from common import *
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
# 0. reference network f (seed 1, real data) -- retrain if absent
d=prep_daystrat("GEF"); sc=d["sdl"].scale_[0]
p1=f"{OUT}/GEF_ours_seed1.pth"
if not os.path.exists(p1): m=make_model(d,0.1,seed=1); t=train(m,d,1,p1); print(f"reference net trained {t:.0f}s",flush=True)
import step4_synthetic as s4        # runs the step-4 experiment (trains syn_p5/p10/p30 and syn_ens0-4 if absent)
S,dd,Xte,sparse=s4.S,s4.dd,s4.Xte,s4.sparse
LOG=open(f"{OUT}/ref_rho_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()
# continuous-covariate norm: lagged load and temperature of the standardized window.
# GEF column order is [hour, load, temperature, month, day of month, day of week], so load = 1, temperature = 2.
cn=np.linalg.norm(Xte[:,:,[1,2]].reshape(len(Xte),-1),axis=1)
f,s=S["te"]["f"],S["te"]["s"]
def row(tag,mu,ale,epi,show_err=True):
    err=np.abs(mu-f)
    log(f"{tag:24s} rho(ale,sigma)={spearmanr(ale,s)[0]:.2f} bias={np.mean(ale/s)-1:+.0%} ratio={epi[sparse].mean()/epi[~sparse].mean():.2f} "
        f"AUC={roc_auc_score(sparse,epi):.2f} rho(epi,|mu-f|)={spearmanr(epi,err)[0]:.2f} rho(epi,||x||)={spearmanr(epi[~sparse],cn[~sparse])[0]:.2f} "
        f"| rho(|mu-f|,||x||)={spearmanr(err[~sparse],cn[~sparse])[0]:.2f}")
log("Table tab:reference recomputed (continuous-covariate norm, 14 dims; Bessel-corrected ensemble variance)")
for p in [0.05,0.10,0.30]:
    m=make_model(dd,p,seed=1); m.load_state_dict(torch.load(f"{OUT}/syn_p{int(p*100)}.pth",weights_only=True)); r=predict(m,Xte)
    row(f"MC dropout p={p:.2f}",r["mu"],r["ale"],r["epi"])
mus,vs=[],[]
for k in range(20):
    m=make_model(dd,0.1,seed=10+k); path=f"{OUT}/syn_ens{k}.pth"
    if os.path.exists(path): m.load_state_dict(torch.load(path,weights_only=True))
    else: t=train(m,dd,10+k,path); print(f"  syn member {k} trained {t:.0f}s",flush=True); m.load_state_dict(torch.load(path,weights_only=True))
    r=predict(m,Xte,1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
mus,vs=np.stack(mus),np.stack(vs)
for K in [5,10,20]:
    row(f"Ensemble K={K}",mus[:K].mean(0),np.sqrt(vs[:K].mean(0)),np.sqrt(mus[:K].var(0,ddof=1)))
# ensemble + dropout (K=5 x 20 masks)
mus2,vs2,within=[],[],[]
for k in range(5):
    m=make_model(dd,0.1,seed=10+k); m.load_state_dict(torch.load(f"{OUT}/syn_ens{k}.pth",weights_only=True)); r=predict(m,Xte,n_mc=20)
    mus2.append(r["mu"]); vs2.append(r["ale"]**2); within.append(r["epi"]**2)
mus2,vs2=np.stack(mus2),np.stack(vs2)
row("Ensemble + dropout",mus2.mean(0),np.sqrt(vs2.mean(0)),np.sqrt(mus2.var(0,ddof=1)+np.mean(within,0)))
# Table tempbins ensemble row (K=5, Bessel) + dropout p=0.1 row for cross-check
zT=Xte[:,-1,2]; bins=[-3,-1,0,0.8,1.5,3]
m=make_model(dd,0.1,seed=1); m.load_state_dict(torch.load(f"{OUT}/syn_p10.pth",weights_only=True)); rd=predict(m,Xte)
e5=np.sqrt(mus[:5].var(0,ddof=1)); err5=np.abs(mus[:5].mean(0)-f)
log("\nTable tab:tempbins (MW): bins <-1 | -1..0 | 0..0.8 | 0.8..1.5 | >1.5")
for name,v in [("Estimation error (ens K=5)",err5),("Known sigma(x)",s),("Dropout epi p=0.1",rd["epi"]),("Ensemble epi K=5 (Bessel)",e5),("Dropout ale",rd["ale"]),("Ensemble ale",np.sqrt(vs[:5].mean(0)))]:
    log(f"{name:28s}"+"".join(f"{v[(zT>=bins[i])&(zT<bins[i+1])].mean()*sc:9.1f}" for i in range(5)))
log(f"{'n':28s}"+"".join(f"{int(((zT>=bins[i])&(zT<bins[i+1])).sum()):9d}" for i in range(5)))
log("DONE")
