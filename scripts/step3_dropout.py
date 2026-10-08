"""Step 3: how much of the epistemic estimate is a dropout-rate artifact?
(1) sweep p, select p on the held-out calibration half by NLL/CRPS;
(2) compare per-input sigma_epi with the K=5 ensemble reference (step 2);
(3) per-input rank stability across p (vs across seeds at fixed p);
(4) verification properties (reducibility, load-decile smile, controlled OOD) at every p."""
from common import *
from scipy.stats import spearmanr, pearsonr
import json
P_LIST=[0.05,0.10,0.20,0.30]; SEED=1
LOG=open(f"{OUT}/step3_log.txt","w")
def log(s): print(s,flush=True); LOG.write(s+"\n"); LOG.flush()

def fit(d, p, seed, tag, frac=1.0, subset_seed=0):
    """Train on a uniformly sampled nested subset (frac) with dropout p."""
    dd=dict(d)
    if frac<1.0:
        rng=np.random.RandomState(subset_seed); perm=rng.permutation(len(d["trl"])); n=int(len(perm)*frac)
        dd["tr"]=d["tr"][np.sort(perm[:n])]; dd["trl"]=d["trl"][np.sort(perm[:n])]
    m=make_model(dd,dropout=p,seed=seed); path=f"{OUT}/{tag}.pth"
    if os.path.exists(path): m.load_state_dict(torch.load(path,weights_only=True))
    else: train(m,dd,seed,path)
    return m

def decile_profile(epi, y, sc, mn):
    Y=y*sc+mn; edges=np.percentile(Y,np.linspace(0,100,11)); prof=[]
    for i in range(10):
        msk=(Y>=edges[i])&(Y<=edges[i+1]) if i==9 else (Y>=edges[i])&(Y<edges[i+1]); prof.append(epi[msk].mean()*sc)
    return np.array(prof)

def run(name):
    log(f"\n{'='*70}\n{name}\n{'='*70}")
    d=prep_daystrat(name); sc,mn=d["sdl"].scale_[0],d["sdl"].mean_[0]
    E=np.load(f"{OUT}/{name}_step2_preds.npz"); ens_epi=E["ens_epi"]; ens_ale=E["ens_ale"]
    log(f"Ensemble (K=5) reference: mean sigma_epi={ens_epi.mean()*sc:.1f}  mean sigma_ale={ens_ale.mean()*sc:.1f}  R_epi={np.mean(ens_epi**2/(ens_epi**2+ens_ale**2)):.3f}")
    epi_by_p={}; res={}
    log(f"\n[1] dropout sweep (seed {SEED}); selection metrics on the held-out calibration half")
    log(f"{'p':>5s} {'valNLL':>8s} {'valCRPS':>8s} {'testMAE':>8s} {'testCRPS':>9s} {'R_epi':>6s} {'epi':>7s} {'ale':>7s} | {'corr(epi,ens)':>13s} {'rho(epi,ens)':>12s} {'scale/ens':>9s} {'corr(epi,|e|)':>13s}")
    for p in P_LIST:
        m=fit(d,p,SEED,f"{name}_p{int(p*100)}_seed{SEED}")
        rc=predict(m,d["cal"]); rt=predict(m,d["te"])
        mc=full_metrics(d["call"],rc["mu"],rc["tot"],d["sdl"]); mt=full_metrics(d["tel"],rt["mu"],rt["tot"],d["sdl"])
        epi_by_p[p]=rt["epi"]; err=np.abs(d["tel"]-rt["mu"])
        r=dict(p=p,valNLL=mc["NLL"],valCRPS=mc["CRPS"],testMAE=mt["MAE"],testCRPS=mt["CRPS"],R_epi=rt["R_epi"],
               epi=rt["epi"].mean()*sc,ale=rt["ale"].mean()*sc,
               corr_ens=pearsonr(rt["epi"],ens_epi)[0],rho_ens=spearmanr(rt["epi"],ens_epi)[0],
               scale_ens=rt["epi"].mean()/ens_epi.mean(),corr_err=pearsonr(rt["epi"],err)[0])
        res[p]=r
        log(f"{p:5.2f} {r['valNLL']:8.4f} {r['valCRPS']:8.2f} {r['testMAE']:8.2f} {r['testCRPS']:9.2f} {r['R_epi']:6.3f} {r['epi']:7.1f} {r['ale']:7.1f} | {r['corr_ens']:13.3f} {r['rho_ens']:12.3f} {r['scale_ens']:9.2f} {r['corr_err']:13.3f}")
    p_nll=min(res,key=lambda p:res[p]["valNLL"]); p_crps=min(res,key=lambda p:res[p]["valCRPS"])
    log(f"selected p: by validation NLL -> {p_nll}; by validation CRPS -> {p_crps}")
    # ensemble-based reference: the p whose epistemic scale is closest to the ensemble's
    p_scale=min(res,key=lambda p:abs(np.log(res[p]["scale_ens"])))
    log(f"p whose mean sigma_epi is closest to the ensemble reference: {p_scale} (ratio {res[p_scale]['scale_ens']:.2f})")

    log(f"\n[2] per-input rank stability of sigma_epi: Spearman rho across p (seed {SEED})")
    ps=P_LIST; M=np.zeros((4,4))
    for i,a in enumerate(ps):
        for j,b in enumerate(ps): M[i,j]=spearmanr(epi_by_p[a],epi_by_p[b])[0]
    log("      "+"".join(f"{b:>8.2f}" for b in ps))
    for i,a in enumerate(ps): log(f"{a:6.2f}"+"".join(f"{M[i,j]:8.3f}" for j in range(4)))
    # across-seed reference at p=0.10
    m2=fit(d,0.10,2,f"{name}_p10_seed2"); e2=predict(m2,d["te"])["epi"]
    rho_seed=spearmanr(epi_by_p[0.10],e2)[0]; rho_seed_ens=spearmanr(e2,ens_epi)[0]
    log(f"reference: rho between two seeds at p=0.10 = {rho_seed:.3f}; seed-2 vs ensemble = {rho_seed_ens:.3f}")
    # stability of the *ranking* is what the flagging policy uses: overlap of top-10% sets
    def top_overlap(a,b,q=0.10):
        ka=a>=np.quantile(a,1-q); kb=b>=np.quantile(b,1-q); return (ka&kb).sum()/ka.sum()
    log("top-10%% overlap: p=0.05 vs 0.30 = %.3f ; p=0.10 vs 0.30 = %.3f ; seed1 vs seed2 (p=0.10) = %.3f ; p=0.10 vs ensemble = %.3f"
        %(top_overlap(epi_by_p[0.05],epi_by_p[0.30]),top_overlap(epi_by_p[0.10],epi_by_p[0.30]),top_overlap(epi_by_p[0.10],e2),top_overlap(epi_by_p[0.10],ens_epi)))

    log(f"\n[3] load-decile profile of mean sigma_epi (MW) at every p, plus the ensemble")
    profs={p:decile_profile(epi_by_p[p],d["tel"],sc,mn) for p in ps}; profs["ens"]=decile_profile(ens_epi,d["tel"],sc,mn)
    log("decile:   "+"".join(f"{k+1:>7d}" for k in range(10))+"   smile? (tails>middle)")
    for k,v in profs.items():
        smile=(v[0]>v[4:6].mean()) and (v[9]>v[4:6].mean())
        log(f"{str(k):>8s}: "+"".join(f"{x:7.1f}" for x in v)+f"   {smile}")
    log("rank correlation of decile profiles with the ensemble profile: "+", ".join(f"p={p}: {spearmanr(profs[p],profs['ens'])[0]:.2f}" for p in ps))

    log(f"\n[4] controlled OOD (input scaling s) at every p: mean sigma_epi (MW) [and sigma_ale]")
    S=[1.0,1.25,1.5,2.0]; log("        "+"".join(f"{'s='+str(s):>16s}" for s in S))
    for p in ps:
        m=fit(d,p,SEED,f"{name}_p{int(p*100)}_seed{SEED}"); row=[]
        for s in S:
            r=predict(m,d["te"]*s); row.append(f"{r['epi'].mean()*sc:7.1f} [{r['ale'].mean()*sc:6.1f}]")
        log(f"p={p:4.2f} "+"".join(f"{x:>16s}" for x in row))

    if name=="GEF":
        log(f"\n[5] reducibility at every p: mean sigma_epi / sigma_ale (MW) on nested uniform subsets 25/50/100%")
        log("        "+"".join(f"{'frac='+str(f):>22s}" for f in [0.25,0.5,1.0]))
        for p in ps:
            row=[]
            for f in [0.25,0.5,1.0]:
                tag=f"{name}_p{int(p*100)}_seed{SEED}" if f==1.0 else f"{name}_p{int(p*100)}_frac{int(f*100)}"
                m=fit(d,p,SEED,tag,frac=f); r=predict(m,d["te"]); mt=full_metrics(d["tel"],r["mu"],r["tot"],d["sdl"])
                row.append(f"{r['epi'].mean()*sc:6.1f} / {r['ale'].mean()*sc:6.1f} (MAE {mt['MAE']:5.1f})")
            log(f"p={p:4.2f} "+"".join(f"{x:>22s}" for x in row))
    np.savez(f"{OUT}/{name}_step3_epi.npz",**{f"p{int(p*100)}":epi_by_p[p] for p in ps},seed2_p10=e2,ens=ens_epi)
    json.dump({str(k):{kk:float(vv) for kk,vv in v.items()} for k,v in res.items()},open(f"{OUT}/{name}_step3.json","w"),indent=1)

if __name__=="__main__":
    for name in ["GEF","COVID"]: run(name)
    log("DONE")
