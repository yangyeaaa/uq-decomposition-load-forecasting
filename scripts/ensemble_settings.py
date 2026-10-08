"""Section 4.4, Supplementary Table S10: does the deep-ensemble epistemic scale depend on training choices that forecast-level
criteria cannot pin down?  K=5 (seeds 10-14), GEFCom2014, same protocol as the paper; vary one knob
at a time: AdamW weight decay, training dropout rate, bootstrap resampling (bagging)."""
import os, sys, json
from common import *
import torch.nn as nn
from scipy.stats import spearmanr
OUTB = f"{OUT}/ens_settings"; os.makedirs(OUTB, exist_ok=True)
K = 5; SEEDS = list(range(10, 10 + K))
CONFIGS = {"base": dict(wd=1e-5, p=0.1, bag=False),
           "wd1e-2": dict(wd=1e-2, p=0.1, bag=False), "wd1e-1": dict(wd=1e-1, p=0.1, bag=False), "wd1": dict(wd=1.0, p=0.1, bag=False),
           "p0": dict(wd=1e-5, p=0.0, bag=False), "p0.3": dict(wd=1e-5, p=0.3, bag=False),
           "bag": dict(wd=1e-5, p=0.1, bag=True)}

def train_cfg(m, d, seed, cfg, path, epochs=50, patience=12):
    MM.setup_seed(seed)
    Xtr, ytr = d["tr"], d["trl"]
    if cfg["bag"]:
        idx = np.random.RandomState(1000 + seed).randint(0, len(Xtr), len(Xtr)); Xtr, ytr = Xtr[idx], ytr[idx]
    tl = torch.utils.data.DataLoader(MM.TimeSeriesDataset(Xtr, ytr), batch_size=256, shuffle=True)
    vl = torch.utils.data.DataLoader(MM.TimeSeriesDataset(d["es"], d["esl"]), batch_size=256, shuffle=False)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5, min_lr=1e-6)
    best, pc = float('inf'), 0
    for ep in range(epochs):
        m.train()
        for x, y in tl:
            y = y.squeeze(); mu, lv, _ = m(x); loss = FX.hetero_nll(y, mu, lv)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval(); vr = []
        with torch.no_grad():
            for x, y in vl:
                mu = m(x)[0]; vr.append(torch.sqrt(((y.squeeze() - mu) ** 2).mean()).item())
        vr = float(np.mean(vr)); sch.step(vr)
        if vr < best: best, pc = vr, 0; torch.save(m.state_dict(), path)
        else:
            pc += 1
            if pc >= patience: break
    m.load_state_dict(torch.load(path, weights_only=True))

def members(name, cfg, d):
    out = {k: ([], []) for k in ["te", "es", "cal"]}
    for s in SEEDS:
        if name == "base" and os.path.exists(f"{OUT}/GEF_ens{s-10}.pth"):
            path = f"{OUT}/GEF_ens{s-10}.pth"   # identical protocol: reuse the K=10 members 0-4
        else:
            path = f"{OUTB}/{name}_s{s}.pth"
        m = make_model(d, cfg["p"], seed=s)
        if not os.path.exists(path): train_cfg(m, d, s, cfg, path)
        m.load_state_dict(torch.load(path, weights_only=True))
        for k in out:
            r = predict(m, d[k], 1, False); out[k][0].append(r["mu"]); out[k][1].append(r["ale"] ** 2)
    res = {}
    for k, (mus, vs) in out.items():
        mus, vs = np.stack(mus), np.stack(vs)
        res[k] = (mus.mean(0), np.sqrt(vs.mean(0)), np.sqrt(mus.var(0, ddof=1)))
    return res

if __name__ == "__main__":
    d = prep_daystrat("GEF"); sdl = d["sdl"]; sc = sdl.scale_[0]
    names = sys.argv[1:] or list(CONFIGS)
    rows = []; ref = None
    for name in names:
        P = members(name, CONFIGS[name], d)
        row = dict(config=name)
        for k, lab in [("es", "es"), ("cal", "cal"), ("te", "test")]:
            mu, ale, epi = P[k]; tot = np.sqrt(ale ** 2 + epi ** 2); y = d[{"es": "esl", "cal": "call", "te": "tel"}[k]]
            m = full_metrics(y, mu, tot, sdl)
            row[f"{lab}_NLL"] = m["NLL"]; row[f"{lab}_CRPS"] = m["CRPS"]
            if k == "te":
                row.update(MAE=m["MAE"], PICP90=m["PICP90"], ECE=m["ECE"], sig_epi=epi.mean() * sc, sig_ale=ale.mean() * sc,
                           R_epi=np.mean(epi ** 2 / tot ** 2))
                np.save(f"{OUTB}/{name}_epi_te.npy", epi)
        rows.append(row)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    base_epi = np.load(f"{OUTB}/base_epi_te.npy") if os.path.exists(f"{OUTB}/base_epi_te.npy") else None
    if base_epi is not None:
        for r in rows:
            r["rho_vs_base"] = spearmanr(np.load(f"{OUTB}/{r['config']}_epi_te.npy"), base_epi)[0]
            print(r["config"], "rank corr of per-input sigma_epi with base:", round(r["rho_vs_base"], 3))
    json.dump(rows, open(f"{OUT}/ensemble_settings_{'_'.join(names)}.json", "w"), indent=1, default=float)
    print("DONE")
