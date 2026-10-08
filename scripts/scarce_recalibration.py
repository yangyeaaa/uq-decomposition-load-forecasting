"""Section 6.3, Supplementary Table S12: data-scarce regime -- does the epistemic term still matter once each interval is recalibrated?
Same protocol as step7/sec63 (first M days, next 14 days split 12h/12h into early-stopping and calibration,
K=5 seeds 10-14, Bessel). For the total and the aleatoric-only scale separately we apply
(i) static split-conformal on the calibration subset and (ii) causal rolling conformal (W=30 days, one-day lag,
initialised with the calibration scores), and compare coverage, width, IS90, ECE and coverage uniformity.
Usage: python scarce_recalibration.py train <GEF|COVID> <M|None> ...  |  analyse"""
import os, sys, json
from common import *
from step7_scarce import make_split
OUTD = f"{OUT}/scarce_recal"; os.makedirs(OUTD, exist_ok=True)
MS = [30, 60, 90, 180, None]; LEVELS10 = np.linspace(0.5, 0.95, 10); B = 1000; BLOCK = 7

def preds(name, M):
    p = f"{OUTD}/{name}_M{M}_preds.npz"
    if os.path.exists(p): return dict(np.load(p))
    d = make_split(name, M); out = {}
    for split in ["te", "cal"]: out[split] = ([], [])
    for k in range(5):
        m = make_model(d, 0.1, seed=10 + k); path = f"{OUTD}/{name}_M{M}_ens{k}.pth"
        if not os.path.exists(path): train(m, d, 10 + k, path)
        m.load_state_dict(torch.load(path, weights_only=True))
        for split in out:
            r = predict(m, d[split], 1, False); out[split][0].append(r["mu"]); out[split][1].append(r["ale"] ** 2)
    R = dict(y=d["tel"], yc=d["call"], scale=d["sdl"].scale_[0], mean=d["sdl"].mean_[0])
    for split, suf in [("te", ""), ("cal", "c")]:
        mus, vs = np.stack(out[split][0]), np.stack(out[split][1])
        R["mu" + suf] = mus.mean(0); R["ale" + suf] = np.sqrt(vs.mean(0)); R["epi" + suf] = np.sqrt(mus.var(0, ddof=1))
    np.savez(p, **R); print("saved", name, M, flush=True); return R

def qhat(s, c):
    n = len(s); k = min(n, int(np.ceil((n + 1) * c))); return np.sort(s)[k - 1]

def multipliers(s_cal, s_te, N, level, mode):
    """per-test-point multiplier; static = calibration quantile; rolling = causal W=30-day window, 1-day lag"""
    if mode == "static": return np.full(N, qhat(s_cal, level))
    days = [np.arange(i, min(i + 24, N)) for i in range(0, N, 24)]; hist = list(s_cal); pending = None; q = np.zeros(N)
    for idx in days:
        q[idx] = qhat(np.array(hist[-30 * 24:]), level)
        if pending is not None: hist += list(pending)
        pending = s_te[idx]
    return q

def mbb(nd, rng):
    out = []
    while len(out) < nd:
        s = rng.randint(0, max(1, nd - BLOCK + 1)); out += list(range(s, min(s + BLOCK, nd)))
    return np.array(out[:nd])

def analyse():
    raw = {n: load_raw(n) for n in ["GEF", "COVID"]}
    MONTHCOL = {"GEF": 3, "COVID": 2}
    rows = []
    for name in ["GEF", "COVID"]:
        div = 1000 if name == "COVID" else 1
        month = raw[name]["test"][0][:, -1, MONTHCOL[name]]
        for M in MS:
            P = preds(name, M); sc = P["scale"] / div
            y, mu, ale, epi = P["y"], P["mu"], P["ale"], P["epi"]; yc, muc, alec, epic = P["yc"], P["muc"], P["alec"], P["epic"]
            tot = np.sqrt(ale ** 2 + epi ** 2); totc = np.sqrt(alec ** 2 + epic ** 2); N = len(y)
            err = np.abs(y - mu); errc = np.abs(yc - muc); Repi = epi ** 2 / tot ** 2
            hiR = Repi >= np.quantile(Repi, 0.8)
            row = dict(data=name, M=M or 359, R_epi=Repi.mean())
            z = norm.ppf(0.95)
            row["raw_cov_tot"] = np.mean(err <= z * tot); row["raw_cov_ale"] = np.mean(err <= z * ale)
            day = np.arange(N) // 24; nd = day.max() + 1
            per = {}
            for scale_name, S, Sc in [("tot", tot, totc), ("ale", ale, alec)]:
                s_cal = errc / Sc; s_te = err / S
                for mode in ["static", "rolling"]:
                    q90 = multipliers(s_cal, s_te, N, 0.90, mode)
                    hw = q90 * S; inside = err <= hw
                    lo_miss = np.maximum(err - hw, 0)
                    is90 = 2 * hw + 2 / 0.1 * lo_miss            # interval score per point (standardised units)
                    obs = [np.mean(err <= multipliers(s_cal, s_te, N, c, mode) * S) for c in LEVELS10]
                    mcov = [inside[month == mm].mean() for mm in np.unique(month)]
                    key = f"{scale_name}_{mode}"
                    per[key] = dict(inside=inside, is90=is90, width=2 * hw)
                    row[f"{key}_cov90"] = inside.mean(); row[f"{key}_width"] = 2 * hw.mean() * sc
                    row[f"{key}_IS90"] = is90.mean() * sc; row[f"{key}_ECE"] = np.mean(np.abs(LEVELS10 - np.array(obs)))
                    row[f"{key}_monthdev"] = np.max(np.abs(np.array(mcov) - 0.9))
                    row[f"{key}_cov_hiR"] = inside[hiR].mean(); row[f"{key}_cov_loR"] = inside[~hiR].mean()
            # paired moving-block bootstrap: IS90(ale) - IS90(tot) (positive = total better), per calibration mode
            rng = np.random.RandomState(0); H = [np.where(day == dd)[0] for dd in range(nd)]
            for mode in ["static", "rolling"]:
                diff = (per[f"ale_{mode}"]["is90"] - per[f"tot_{mode}"]["is90"]) * sc
                dcov = per[f"tot_{mode}"]["inside"].astype(float) - per[f"ale_{mode}"]["inside"]
                bs, bc = [], []
                for _ in range(B):
                    ii = np.concatenate([H[j] for j in mbb(nd, rng)]); bs.append(diff[ii].mean()); bc.append(dcov[ii].mean())
                row[f"dIS90_{mode}"] = diff.mean(); row[f"dIS90_{mode}_lo"], row[f"dIS90_{mode}_hi"] = np.percentile(bs, [2.5, 97.5])
                row[f"dcov_{mode}"] = dcov.mean(); row[f"dcov_{mode}_lo"], row[f"dcov_{mode}_hi"] = np.percentile(bc, [2.5, 97.5])
            # coverage among the 20 % of inputs with the largest R_epi (rolling calibration), paired block bootstrap
            dh = per["tot_rolling"]["inside"][hiR].mean() - per["ale_rolling"]["inside"][hiR].mean(); bh = []; rng_h = np.random.RandomState(0)
            for _ in range(B):
                ii = np.concatenate([H[j] for j in mbb(nd, rng_h)]); h = hiR[ii]
                bh.append(per["tot_rolling"]["inside"][ii][h].mean() - per["ale_rolling"]["inside"][ii][h].mean())
            row["dcov_hiR_rolling"] = dh; row["dcov_hiR_rolling_lo"], row["dcov_hiR_rolling_hi"] = np.percentile(bh, [2.5, 97.5])
            rows.append(row)
            print(f"{name} M={row['M']}: rolling PICP90 total/ale {row['tot_rolling_cov90']:.3f}/{row['ale_rolling_cov90']:.3f}  "
                  f"dIS90 {row['dIS90_rolling']:+.1f} [{row['dIS90_rolling_lo']:+.1f},{row['dIS90_rolling_hi']:+.1f}]  "
                  f"high-R_epi coverage {row['tot_rolling_cov_hiR']:.3f}/{row['ale_rolling_cov_hiR']:.3f} "
                  f"diff {dh:+.3f} [{row['dcov_hiR_rolling_lo']:+.3f},{row['dcov_hiR_rolling_hi']:+.3f}]", flush=True)
    json.dump(rows, open(f"{OUT}/scarce_recalibration.json", "w"), indent=1, default=float)

if __name__ == "__main__":
    if sys.argv[1] == "train":
        name = sys.argv[2]
        for a in sys.argv[3:]: preds(name, None if a == "None" else int(a))
    else:
        analyse()
