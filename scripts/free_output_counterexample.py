"""Table 1(a) and Fig. 2(a,b): the free-output construction (Section 2.2).

Two dispersion heads (aleatoric, epistemic) are emitted separately and enter a single Gaussian NLL
only through their summed variance. Only the initial log-scales of the two heads are changed; data,
architecture, optimizer, seed, and early stopping are fixed, and no component is supervised directly.

Usage:  python free_output_counterexample.py [i ...]     (i indexes INITS; default: all three)
"""
from common import *
import sys
import torch.nn as nn

INITS = [(0.0, -1.0), (-1.0, 0.0), (-2.0, 0.5)]      # (log s_ale, log s_epi)
SEED = 1


def build(d, log_ale, log_epi):
    MM.setup_seed(SEED)
    m = MM.DiffLoadV3(d["sl"], d["nv"], 64, 4, 2, 5, dropout=0.1)
    with torch.no_grad():
        m.uncertainty_head.log_aleatoric_scale.fill_(log_ale)
        m.uncertainty_head.log_epistemic_scale.fill_(log_epi)
    return m


def nll(y, out):
    pred = out["main_value"] + out["residual_mu"]
    var = out["aleatoric"] ** 2 + out["epistemic"] ** 2 + 1e-6
    return (0.5 * (torch.log(var) + (y - pred) ** 2 / var)).mean()


def fit(m, d, path, epochs=50, patience=12):
    MM.setup_seed(SEED); tl, vl = loaders(d)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5, min_lr=1e-6)
    best, pc = float("inf"), 0
    for ep in range(epochs):
        m.train()
        for x, y in tl:
            loss = nll(y.squeeze(), m(x, use_diffusion=False))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval(); vr = []
        with torch.no_grad():
            for x, y in vl:
                o = m(x, use_diffusion=False)
                vr.append(torch.sqrt(((y.squeeze() - o["main_value"] - o["residual_mu"]) ** 2).mean()).item())
        vr = float(np.mean(vr)); sch.step(vr)
        if vr < best:
            best, pc = vr, 0; torch.save(m.state_dict(), path)
        else:
            pc += 1
            if pc >= patience: break
    m.load_state_dict(torch.load(path, weights_only=True))


@torch.no_grad()
def evaluate(m, d):
    m.eval(); o = m(torch.tensor(d["te"], dtype=torch.float32), use_diffusion=False)
    mu = (o["main_value"] + o["residual_mu"]).numpy()
    a2 = o["aleatoric"].numpy() ** 2; e2 = o["epistemic"].numpy() ** 2
    tot = np.sqrt(a2 + e2); mm = full_metrics(d["tel"], mu, tot, d["sdl"])
    return dict(R_epi=float(np.mean(e2 / (a2 + e2))), sig_total=float(tot.mean()), **mm)


if __name__ == "__main__":
    d = prep_daystrat("GEF")
    sel = [INITS[int(i)] for i in sys.argv[1:]] or INITS
    LOG = open(f"{OUT}/free_output_log_{'_'.join(sys.argv[1:]) or 'all'}.txt", "w")
    for la, le in sel:
        path = f"{OUT}/GEF_freeoutput_{la:+.1f}_{le:+.1f}.pth"; m = build(d, la, le)
        if not os.path.exists(path): fit(m, d, path)
        else: m.load_state_dict(torch.load(path, weights_only=True))
        r = evaluate(m, d)
        line = (f"init (log s_ale, log s_epi) = ({la:+.1f}, {le:+.1f}): R_epi={r['R_epi']:.3f} "
                f"sigma_total={r['sig_total']:.4f} MAE={r['MAE']:.1f} ECE={r['ECE']:.3f} PICP90={r['PICP90']:.3f}")
        print(line, flush=True); LOG.write(line + "\n")
