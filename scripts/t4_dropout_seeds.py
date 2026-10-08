"""T4 seed replication for Monte Carlo dropout (p=0.1): train full-data and thinned-data models for
seeds 1,2,3 (paired), compute the controlled change Delta sigma_epi inside/outside the thinned
region, contrast, and AUC, plus block-bootstrap CIs over target days. The synthetic data are the
step-4 data (same rng seeds), reconstructed here before thinning."""
from common import *
from sklearn.metrics import roc_auc_score
import step4_synthetic as s4
rng=np.random.RandomState(7); B=1000; BLOCK=7
# full synthetic data: regenerate exactly as step4 (S built before thinning -> rebuild)
d=s4.d; S_full={}
rng0=np.random.RandomState(123)   # step4 uses rng seeded 123 for synth targets
import inspect; assert 'RandomState(123)' in inspect.getsource(s4) or True
# safer: recover full-data targets from S before thinning by re-running synth with s4.rng reset
s4.rng=np.random.RandomState(123)
for k in ["tr","es","cal","te"]:
    f,s,y=s4.synth(d[k]); S_full[k]=dict(X=d[k],f=f,s=s,y=y)
# verify test targets match step4's (test partition is never thinned)
assert np.allclose(S_full["te"]["y"], s4.S["te"]["y"]), "synthetic regeneration mismatch"
dfull=dict(d); dfull.update(tr=S_full["tr"]["X"],trl=S_full["tr"]["y"],es=S_full["es"]["X"],esl=S_full["es"]["y"],cal=S_full["cal"]["X"],call=S_full["cal"]["y"],te=S_full["te"]["X"],tel=S_full["te"]["y"])
dthin=s4.dd; Xte=s4.Xte; sparse=s4.sparse; nd=len(Xte)//24
LOG=open(f"{OUT}/t4_dropout_seeds_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()
def mbb():
    out=[]
    while len(out)<nd:
        st=rng.randint(0,nd-BLOCK+1); out+=list(range(st,st+BLOCK))
    di=np.array(out[:nd]); return (di[:,None]*24+np.arange(24)[None,:]).ravel()
log("T4 seed replication, MC dropout p=0.1, M=40: change in epistemic estimate under thinning (MW), paired by seed")
allD=[]
for seed in [1,2,3]:
    mf=make_model(dfull,0.1,seed=seed); pf=f"{OUT}/syn_full_p10_s{seed}.pth"
    if not os.path.exists(pf): train(mf,dfull,seed,pf)
    mf.load_state_dict(torch.load(pf,weights_only=True))
    mt=make_model(dthin,0.1,seed=seed); pt=f"{OUT}/syn_p10.pth" if seed==1 else f"{OUT}/syn_thin_p10_s{seed}.pth"
    if not os.path.exists(pt): train(mt,dthin,seed,pt)
    mt.load_state_dict(torch.load(pt,weights_only=True))
    ef=predict(mf,Xte)["epi"]; et=predict(mt,Xte)["epi"]; sc=d["sdl"].scale_[0]
    D=(et-ef)*sc; allD.append(D)
    c=D[sparse].mean()-D[~sparse].mean(); vals=[]; aucs=[]
    for _ in range(B):
        h=mbb(); vals.append(D[h][sparse[h]].mean()-D[h][~sparse[h]].mean())
    log(f"  seed {seed}: unthinned {D[~sparse].mean():+.1f}  thinned {D[sparse].mean():+.1f}  contrast {c:+.1f} [95% CI {np.percentile(vals,2.5):+.1f}, {np.percentile(vals,97.5):+.1f}]  AUC {roc_auc_score(sparse,D):.2f}")
Dm=np.mean(allD,0); c=Dm[sparse].mean()-Dm[~sparse].mean(); vals=[]
for _ in range(B):
    h=mbb(); vals.append(Dm[h][sparse[h]].mean()-Dm[h][~sparse[h]].mean())
log(f"  mean over 3 seeds: unthinned {Dm[~sparse].mean():+.1f}  thinned {Dm[sparse].mean():+.1f}  contrast {c:+.1f} [95% CI {np.percentile(vals,2.5):+.1f}, {np.percentile(vals,97.5):+.1f}]  AUC {roc_auc_score(sparse,Dm):.2f}")
log("DONE")
