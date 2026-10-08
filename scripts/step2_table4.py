"""Step 2: disjoint early-stopping / conformal-calibration split, canonical model,
all baselines with proper scoring rules and timings. Saves predictions for later steps."""
from common import *
import torch.nn as nn, json, sys
sys.path.insert(0, f"{ROOT}/src")
import quantile_regression_baseline as QR

SEED = 1
K_ENS = 5
LOG = open(f"{OUT}/step2_log.txt", "w")
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# prep_daystrat is provided by common.py (SPLIT env selects hour- or day-based split)

# ---------- classical MC dropout (homoscedastic, mean head only, MSE) ----------
class MeanOnly(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.encoder = MM.iTransformerEncoder(d["sl"], d["nv"], 64, 4, 2, dropout)
        self.shared = nn.Sequential(nn.Linear(64,64), nn.GELU(), nn.Dropout(dropout))
        self.mu = nn.Linear(64,1)
    def forward(self, x): return self.mu(self.shared(self.encoder(x))).squeeze(-1)

def train_mse(model, d, seed, save, epochs=50, patience=12):
    MM.setup_seed(seed); tl, vl = loaders(d); model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5, min_lr=1e-6)
    best = float('inf'); pc = 0; t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x,y in tl:
            loss = ((model(x)-y.squeeze())**2).mean(); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); vr=[]
        with torch.no_grad():
            for x,y in vl: vr.append(torch.sqrt(((model(x)-y.squeeze())**2).mean()).item())
        vr=float(np.mean(vr)); sch.step(vr)
        if vr<best: best=vr; pc=0; torch.save(model.state_dict(), save)
        else:
            pc+=1
            if pc>=patience: break
    model.load_state_dict(torch.load(save, weights_only=True)); return time.time()-t0

@torch.no_grad()
def mcd_predict(model, X, n_mc=40):
    FX.enable_mc_dropout(model); X = torch.tensor(X, dtype=torch.float32)
    mus = np.stack([model(X).numpy() for _ in range(n_mc)], 0)
    return mus.mean(0), mus.var(0)

def train_qr_timed(d, seed, save):
    MM.setup_seed(seed); tl, vl = loaders(d)
    MM.setup_seed(seed); m = QR.QuantileNet(d["sl"], d["nv"]); t0 = time.time()
    QR.train_qr(m, tl, vl, epochs=50, patience=12, save=save); return m, time.time()-t0

def qr_metrics(Q, y, sdl):
    """Q: (N,99) sorted quantiles in standardized units. Metrics comparable to full_metrics."""
    sc,mn = sdl.scale_[0], sdl.mean_[0]; Y = y*sc+mn; Q = Q*sc+mn; qs = QR.QGRID
    med = Q[:, 49]; e = Y-med
    m = dict(MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e**2)), MAPE=100*np.mean(np.abs(e/Y)))
    pin_all = np.array([np.mean(np.maximum(q*(Y-Q[:,i]), (q-1)*(Y-Q[:,i]))) for i,q in enumerate(qs)])
    m["CRPS"] = 2*np.mean(pin_all)             # CRPS = 2 * integral of pinball over tau (dense grid approx.)
    m["NLL"] = np.nan
    m["Pinball"] = np.mean(pin_all[4::5][:19])  # levels 0.05,0.10,...,0.95 to match the Gaussian models
    lo, hi = Q[:,4], Q[:,94]; a=0.1
    m["IS90"] = np.mean((hi-lo)+2/a*(lo-Y)*(Y<lo)+2/a*(Y-hi)*(Y>hi))
    obs=[]
    for c in LEVELS:
        il = int(round((1-c)/2*100))-1; ih = int(round((1+c)/2*100))-1
        obs.append(np.mean((Y>=Q[:,il])&(Y<=Q[:,ih])))
    obs=np.array(obs); m["ECE"]=np.mean(np.abs(LEVELS-obs)); m["PICP90"]=obs[-2]
    return m, obs

def cov_curve(y, mu, sig, sdl, mult=None):
    sc,mn=sdl.scale_[0],sdl.mean_[0]; Y=y*sc+mn; P=mu*sc+mn; S=sig*sc
    return np.array([np.mean(np.abs(Y-P)<=(norm.ppf((1+c)/2) if mult is None else mult[i])*S) for i,c in enumerate(LEVELS)])

def run(name):
    log(f"\n{'='*70}\n{name}\n{'='*70}")
    d = prep_daystrat(name); sdl = d["sdl"]; sc = sdl.scale_[0]
    log(f"train={len(d['trl'])}  early-stop={d['n_es']}  calibration={d['n_cal']}  test={len(d['tel'])}")
    rows = {}; curves = {}; preds = {}; times = {}

    # ---- Ours: heteroscedastic + MC dropout ----
    model = make_model(d, dropout=0.1, seed=SEED)
    times["Ours"] = dict(train=train(model, d, SEED, f"{OUT}/{name}_ours_seed{SEED}.pth"))
    t0=time.time(); rt = predict(model, d["te"], n_mc=40); times["Ours"]["infer40"] = time.time()-t0
    t0=time.time(); _ = predict(model, d["te"], n_mc=20); times["Ours"]["infer20"] = time.time()-t0
    rcal = predict(model, d["cal"], n_mc=40)
    rows["Ours (raw)"] = full_metrics(d["tel"], rt["mu"], rt["tot"], sdl)
    # temperature scaling on the calibration half
    tau = float(np.sqrt(np.mean(((d["call"]-rcal["mu"])/rcal["tot"])**2)))
    rows["Ours (+temperature)"] = full_metrics(d["tel"], rt["mu"], rt["tot"]*tau, sdl)
    cm = conformal_metrics(d["call"], rcal["mu"], rcal["tot"], d["tel"], rt["mu"], rt["tot"], sdl)
    rows["Ours (+conformal)"] = {**{k:v for k,v in rows["Ours (raw)"].items() if k in ("MAE","RMSE","MAPE","CRPS","NLL","Pinball")},
                                 "IS90": cm["IS90_conf"], "ECE": cm["ECE_conf"], "PICP90": cm["PICP90_conf"]}
    qmult = [conformal_q(d["call"], rcal["mu"], rcal["tot"], c) for c in LEVELS]
    curves["raw"] = cov_curve(d["tel"], rt["mu"], rt["tot"], sdl)
    curves["temp"] = cov_curve(d["tel"], rt["mu"], rt["tot"]*tau, sdl)
    curves["conf"] = cov_curve(d["tel"], rt["mu"], rt["tot"], sdl, mult=qmult)
    preds["ours"] = dict(mu=rt["mu"], ale=rt["ale"], epi=rt["epi"], mu_cal=rcal["mu"], tot_cal=rcal["tot"], ale_cal=rcal["ale"], epi_cal=rcal["epi"])
    log(f"Ours: R_epi={rt['R_epi']:.3f} epi={rt['epi'].mean()*sc:.1f} ale={rt['ale'].mean()*sc:.1f} tau={tau:.3f} q90={cm['q90']:.3f}")

    # ---- MVE: same trained heteroscedastic network, deterministic ----
    t0=time.time(); rd = predict(model, d["te"], n_mc=1, mc=False); times["MVE"] = dict(train=times["Ours"]["train"], infer=time.time()-t0)
    rows["MVE (heteroscedastic, deterministic)"] = full_metrics(d["tel"], rd["mu"], rd["ale"], sdl)
    preds["mve"] = dict(mu=rd["mu"], ale=rd["ale"])

    # ---- classical MC dropout: homoscedastic ----
    MM.setup_seed(SEED); mcd = MeanOnly(d, 0.1)
    times["MC-Dropout"] = dict(train=train_mse(mcd, d, SEED, f"{OUT}/{name}_mcd_seed{SEED}.pth"))
    t0=time.time(); mu_t, ve_t = mcd_predict(mcd, d["te"]); times["MC-Dropout"]["infer40"] = time.time()-t0
    mu_e, ve_e = mcd_predict(mcd, d["es"])
    s2 = max(np.mean((d["esl"]-mu_e)**2) - np.mean(ve_e), 1e-6)   # tau^-1: residual variance minus epistemic, on early-stop half
    rows["MC-Dropout (homoscedastic)"] = full_metrics(d["tel"], mu_t, np.sqrt(ve_t + s2), sdl)
    preds["mcd"] = dict(mu=mu_t, epi=np.sqrt(ve_t), noise=np.sqrt(s2))
    log(f"MC-Dropout: epi={np.sqrt(ve_t).mean()*sc:.1f} noise={np.sqrt(s2)*sc:.1f} R_epi={np.mean(ve_t/(ve_t+s2)):.3f}")

    # ---- deep ensemble K=5 (heteroscedastic members, deterministic, mixture) ----
    mus, vs, ttot = [], [], 0.0
    for k in range(K_ENS):
        m = make_model(d, 0.1, seed=10+k); ttot += train(m, d, 10+k, f"{OUT}/{name}_ens{k}.pth")
        r = predict(m, d["te"], n_mc=1, mc=False); mus.append(r["mu"]); vs.append(r["ale"]**2)
        log(f"  ensemble member {k+1}/{K_ENS} done")
    mus, vs = np.stack(mus), np.stack(vs)
    mu_s = mus.mean(0); v_epi = mus.var(0); v_ale = vs.mean(0)
    times["Deep Ensemble (K=5)"] = dict(train=ttot, infer=K_ENS*times["MVE"]["infer"])
    rows["Deep Ensemble (K=5)"] = full_metrics(d["tel"], mu_s, np.sqrt(v_ale+v_epi), sdl)
    preds["ens"] = dict(mu=mu_s, ale=np.sqrt(v_ale), epi=np.sqrt(v_epi), members_mu=mus)
    log(f"Ensemble: R_epi={np.mean(v_epi/(v_epi+v_ale)):.3f} epi={np.sqrt(v_epi).mean()*sc:.1f} ale={np.sqrt(v_ale).mean()*sc:.1f}")

    # ---- quantile regression ----
    qr, tq = train_qr_timed(d, SEED, f"{OUT}/{name}_qr_seed{SEED}.pth"); qr.eval()
    with torch.no_grad():
        t0=time.time(); Q = np.sort(qr(torch.tensor(d["te"],dtype=torch.float32)).numpy(),1); ti=time.time()-t0
    times["Quantile Regression"] = dict(train=tq, infer=ti)
    rows["Quantile Regression (median for MAE)"], curves["qr"] = qr_metrics(Q, d["tel"], sdl)
    preds["qr"] = dict(Q=Q)

    # ---- report ----
    keys = ["MAE","RMSE","MAPE","CRPS","NLL","Pinball","IS90","ECE","PICP90"]
    log("\n" + f"{'Method':40s}" + "".join(f"{k:>10s}" for k in keys))
    for r,m in rows.items(): log(f"{r:40s}" + "".join(f"{m[k]:>10.4g}" if not np.isnan(m[k]) else f"{'—':>10s}" for k in keys))
    log("\nTimings (s):")
    for r,t in times.items(): log(f"  {r:22s} " + " ".join(f"{k}={v:.1f}" for k,v in t.items()))
    np.savez(f"{OUT}/{name}_step2_preds.npz", tel=d["tel"], call=d["call"], scale=sc, mean=sdl.mean_[0],
             **{f"{m}_{k}": v for m,dd in preds.items() for k,v in dd.items()})
    json.dump(dict(rows={r:{k:float(v) for k,v in m.items()} for r,m in rows.items()},
                   times=times, curves={k:list(map(float,v)) for k,v in curves.items()},
                   n=dict(train=len(d['trl']), es=d['n_es'], cal=d['n_cal'], test=len(d['tel']))),
              open(f"{OUT}/{name}_step2.json","w"), indent=1)

if __name__ == "__main__":
    log(f"torch {torch.__version__}, CPU threads {torch.get_num_threads()}, {os.cpu_count()} cores")
    for name in ["GEF","COVID"]: run(name)
    log("DONE")
