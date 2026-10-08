"""Experiment c: reducibility with replicates (GEF, ensemble K=5, fractions x 3 replicates).
Experiment d: OOD probes that replace the 'scale all inputs' probe:
   (d1) held-out temperature band: train without z_T>1.0, compare in-band vs out-of-band test points;
   (d2) temperature-only shift of the last lag day by +delta std (within observed range), full-data K=10 ensemble."""
from common import *
from scipy.stats import spearmanr
import json
LOG=open(f"{OUT}/expcd_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()
T_COL=2; FR=[0.10,0.25,0.50,0.75,1.0]; REPS=3; K=5
d=prep_daystrat("GEF"); sc,mn=d["sdl"].scale_[0],d["sdl"].mean_[0]; Xte=d["te"]; y=d["tel"]

def ens(tag,dd,X,seeds):
    mus,vs=[],[]
    for k,sd in enumerate(seeds):
        m=make_model(dd,0.1,seed=sd); path=f"{OUT}/{tag}_m{k}.pth"
        if os.path.exists(path): m.load_state_dict(torch.load(path,weights_only=True))
        else: train(m,dd,sd,path)
        r=predict(m,X,1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
    mus,vs=np.stack(mus),np.stack(vs); return mus.mean(0),np.sqrt(vs.mean(0)),np.sqrt(mus.var(0))

def members10(name,dd,X,KK):
    mus,vs=[],[]
    for k in range(KK):
        m=make_model(dd,0.1); m.load_state_dict(torch.load(f"{OUT}/{name}_ens{k}.pth",weights_only=True))
        r=predict(m,X,1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
    mus,vs=np.stack(mus),np.stack(vs); return mus.mean(0),np.sqrt(vs.mean(0)),np.sqrt(mus.var(0))
log("="*70+"\n[c] reducibility: nested uniform subsets, 3 replicates, K=5 ensemble, GEF\n"+"="*70)
Y=y*sc+mn; dec_lo=Y<=np.percentile(Y,10); dec_hi=Y>=np.percentile(Y,90)
res={f:[] for f in FR}
for r in range(REPS):
    perm=np.random.RandomState(1000+r).permutation(len(d["trl"]))
    for f in FR:
        n=int(len(perm)*f); idx=np.sort(perm[:n]); dd=dict(d); dd["tr"]=d["tr"][idx]; dd["trl"]=d["trl"][idx]
        mu,ale,epi=ens(f"red_r{r}_f{int(f*100)}",dd,Xte,[100*r+10+k for k in range(K)])
        tot=np.sqrt(ale**2+epi**2); m=full_metrics(y,mu,tot,d["sdl"])
        res[f].append(dict(epi=epi.mean()*sc,ale=ale.mean()*sc,R=np.mean(epi**2/tot**2),MAE=m["MAE"],CRPS=m["CRPS"],
                           epi_tails=epi[dec_lo|dec_hi].mean()*sc,epi_mid=epi[~(dec_lo|dec_hi)].mean()*sc,PICP_ale=np.mean(np.abs(y-mu)<=norm.ppf(.95)*ale),PICP_tot=m["PICP90"]))
        log(f"  rep {r} frac {f:.2f} n={n}: epi={res[f][-1]['epi']:.1f} ale={res[f][-1]['ale']:.1f} R={res[f][-1]['R']:.3f} MAE={m['MAE']:.1f}")
log(f"\n{'frac':>6s}{'n':>6s}{'sig_epi':>16s}{'sig_ale':>16s}{'R_epi':>16s}{'MAE':>14s}{'CRPS':>14s}{'epi tails':>12s}{'epi mid':>10s}{'PICP ale-only':>14s}{'PICP total':>12s}")
for f in FR:
    A={k:np.array([x[k] for x in res[f]]) for k in res[f][0]}
    ms=lambda k,fmt: f"{A[k].mean():{fmt}} ± {A[k].std(ddof=1):{fmt}}"
    log(f"{f:6.2f}{int(len(d['trl'])*f):6d}{ms('epi','.1f'):>16s}{ms('ale','.1f'):>16s}{ms('R','.3f'):>16s}{ms('MAE','.1f'):>14s}{ms('CRPS','.1f'):>14s}{A['epi_tails'].mean():12.1f}{A['epi_mid'].mean():10.1f}{A['PICP_ale'].mean():14.3f}{A['PICP_tot'].mean():12.3f}")
# monotonicity check per replicate
for r in range(REPS):
    e=[res[f][r]["epi"] for f in FR]; log(f"  replicate {r}: sigma_epi by fraction = {np.round(e,1).tolist()}  monotone decreasing: {all(np.diff(e)<=0)}")
json.dump({str(f):res[f] for f in FR},open(f"{OUT}/expc.json","w"),indent=1,default=float)

log("\n"+"="*70+"\n[d1] held-out temperature band: train without last-lag z_T>1.0 (K=5), test in-band vs out-of-band\n"+"="*70)
zT_tr=d["tr"][:,-1,T_COL]; keep=zT_tr<=1.0; dd=dict(d); dd["tr"]=d["tr"][keep]; dd["trl"]=d["trl"][keep]
ke=d["es"][:,-1,T_COL]<=1.0; dd["es"]=d["es"][ke]; dd["esl"]=d["esl"][ke]
log(f"training points kept: {keep.sum()}/{len(keep)}")
mu,ale,epi=ens("band",dd,Xte,[10+k for k in range(K)]); zT=Xte[:,-1,T_COL]; inb=zT>1.0
mu0,ale0,epi0=members10("GEF",d,Xte,K)   # full-data reference, same seeds
err=np.abs(y-mu); err0=np.abs(y-mu0)
log(f"{'':28s}{'out-of-band (seen)':>20s}{'in-band (unseen)':>18s}{'ratio':>8s}")
for lab,v in [("sigma_epi, band-holdout",epi),("sigma_epi, full-data",epi0),("sigma_ale, band-holdout",ale),("sigma_ale, full-data",ale0),("|error|, band-holdout",err),("|error|, full-data",err0)]:
    log(f"{lab:28s}{v[~inb].mean()*sc:20.1f}{v[inb].mean()*sc:18.1f}{v[inb].mean()/v[~inb].mean():8.2f}")
log(f"n out-of-band={int((~inb).sum())}, in-band={int(inb.sum())}")
log(f"PICP90 in-band: ale-only={np.mean(err[inb]<=norm.ppf(.95)*ale[inb]):.3f}  total={np.mean(err[inb]<=norm.ppf(.95)*np.sqrt(ale[inb]**2+epi[inb]**2)):.3f} | out-of-band: ale-only={np.mean(err[~inb]<=norm.ppf(.95)*ale[~inb]):.3f} total={np.mean(err[~inb]<=norm.ppf(.95)*np.sqrt(ale[~inb]**2+epi[~inb]**2)):.3f}")

log("\n"+"="*70+"\n[d2] temperature-only shift of the last lag day (+delta std), full-data K=10 ensemble\n"+"="*70)
log(f"{'delta':>6s}{'sig_epi':>9s}{'sig_ale':>9s}{'R_epi':>7s}{'frac of shifted temps beyond train max':>40s}")
tmax=d["tr"][:,-1,T_COL].max()
for delta in [0,0.5,1.0,1.5,2.0]:
    X=Xte.copy(); X[:,-1,T_COL]+=delta
    mu,ale,epi=members10("GEF",d,X,10); log(f"{delta:6.1f}{epi.mean()*sc:9.1f}{ale.mean()*sc:9.1f}{np.mean(epi**2/(epi**2+ale**2)):7.3f}{np.mean(X[:,-1,T_COL]>tmax):40.3f}")
log("(reference: original 'scale all inputs' probe, MC-dropout p=0.1: epi 38.6->54.5, ale 155.8->248.5 for s=1->2)")
log("DONE")
