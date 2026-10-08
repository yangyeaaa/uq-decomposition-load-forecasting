"""T1 (Section 4.1) and the seed band of Fig. 2(a): five training seeds of the heteroscedastic backbone
(dropout p=0.1) under the main protocol.

For each seed the network is evaluated
  - deterministically (MVE), giving the MAE used for the +-1 s.d. reference band in Fig. 2(a), and
  - with Monte Carlo dropout (M=40 passes, Bessel-corrected), giving the T1 epistemic share R_epi.
Seed 1 is the network of Table 4 (rows "MVE" and the dropout estimator). MC dropout masks are drawn
without a fixed seed, so R_epi can differ in the fourth decimal between runs.

Usage:  python t1_seed_stability.py [GEF|COVID ...]      (default: GEF)
"""
from common import *

SEEDS = [1, 2, 3, 4, 5]
M = 40


def run(name):
    d = prep_daystrat(name); sdl = d["sdl"]
    LOG = open(f"{OUT}/t1_seed_stability_{name}_log.txt", "w")
    def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    log(f"{name}: heteroscedastic backbone, p=0.1, seeds {SEEDS}; MC dropout M={M} (Bessel-corrected)")
    rows = []
    for s in SEEDS:
        path = f"{OUT}/{name}_ours_seed{s}.pth"              # seed 1 is shared with step2_table4.py
        m = make_model(d, dropout=0.1, seed=s)
        if os.path.exists(path): m.load_state_dict(torch.load(path, weights_only=True))
        else: train(m, d, s, path)
        det = predict(m, d["te"], n_mc=1, mc=False)
        mve = full_metrics(d["tel"], det["mu"], det["tot"], sdl)
        mc = predict(m, d["te"], n_mc=M, mc=True)
        epi = mc["epi"] * np.sqrt(M / (M - 1)); tot = np.sqrt(mc["ale"] ** 2 + epi ** 2)
        R = float(np.mean(epi ** 2 / tot ** 2))
        mcm = full_metrics(d["tel"], mc["mu"], tot, sdl)
        sig = float(det["tot"].mean())                         # mean predictive s.d., standardized units
        rows.append((s, mve["MAE"], R, sig, mve["ECE"], mve["PICP90"]))
        log(f"seed {s}: MVE MAE={mve['MAE']:.4g} sigma_total={sig:.4f} ECE={mve['ECE']:.4f} "
            f"PICP90={mve['PICP90']:.4f} | MC dropout R_epi={R:.4f} MAE={mcm['MAE']:.4g} "
            f"ECE={mcm['ECE']:.4f} PICP90={mcm['PICP90']:.4f}")
    A = np.array([r[1:] for r in rows])
    for j, lab in enumerate(["MVE MAE", "MC dropout R_epi", "MVE sigma_total", "MVE ECE", "MVE PICP90"]):
        log(f"{lab}: mean={A[:,j].mean():.4g} sd={A[:,j].std(ddof=1):.4g} "
            f"range=[{A[:,j].min():.4g},{A[:,j].max():.4g}]")
    log("DONE")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["GEF"]):
        run(name)
