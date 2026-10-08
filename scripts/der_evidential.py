"""Table 1(b), Fig. 2(c), Supplementary Table S9: deep evidential regression (Amini et al. 2020) on the paper's backbone and protocol.
NIG head (gamma, nu, alpha, beta); predictive = Student-t(gamma, beta(1+nu)/(nu*alpha), 2alpha).
aleatoric = beta/(alpha-1), epistemic = beta/(nu(alpha-1)), so R_epi = 1/(1+nu) per input.
The NLL depends on (nu, beta) only through beta(1+nu)/nu -> one direction left free by the likelihood.
Sweep the evidence-regulariser weight lambda and seeds; report forecast quality and R_epi."""
import os, json
from common import *
import torch.nn as nn, torch.nn.functional as F
from scipy.stats import t as tdist
OUTA = f"{OUT}/der"; os.makedirs(OUTA, exist_ok=True)

class DER(nn.Module):
    def __init__(self, sl, nv, d=64, dropout=0.1):
        super().__init__()
        self.encoder = MM.iTransformerEncoder(sl, nv, d, 4, 2, dropout)
        self.shared = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(d, 4)
    def forward(self, x):
        o = self.out(self.shared(self.encoder(x)))
        g = o[:, 0]; nu = F.softplus(o[:, 1]) + 1e-6; a = F.softplus(o[:, 2]) + 1.0 + 1e-6; b = F.softplus(o[:, 3]) + 1e-6
        return g, nu, a, b

def nig_nll(y, g, nu, a, b):
    om = 2 * b * (1 + nu)
    return (0.5 * torch.log(np.pi / nu) - a * torch.log(om) + (a + 0.5) * torch.log(nu * (y - g) ** 2 + om)
            + torch.lgamma(a) - torch.lgamma(a + 0.5))

def train_der(m, d, seed, lam, path, epochs=50, patience=12):
    MM.setup_seed(seed); tl, vl = loaders(d)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5, min_lr=1e-6)
    best, pc = float('inf'), 0
    for ep in range(epochs):
        m.train()
        for x, y in tl:
            y = y.squeeze(); g, nu, a, b = m(x)
            loss = (nig_nll(y, g, nu, a, b) + lam * torch.abs(y - g) * (2 * nu + a)).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval(); vr = []
        with torch.no_grad():
            for x, y in vl:
                g = m(x)[0]; vr.append(torch.sqrt(((y.squeeze() - g) ** 2).mean()).item())
        vr = float(np.mean(vr)); sch.step(vr)
        if vr < best: best, pc = vr, 0; torch.save(m.state_dict(), path)
        else:
            pc += 1
            if pc >= patience: break
    m.load_state_dict(torch.load(path, weights_only=True)); return best

@torch.no_grad()
def pred(m, X):
    m.eval(); g, nu, a, b = [v.numpy().astype(np.float64) for v in m(torch.tensor(X, dtype=torch.float32))]
    return g, nu, a, b

def scores(y, g, nu, a, b, sdl):
    sc, mn = sdl.scale_[0], sdl.mean_[0]
    Y = y * sc + mn; P = g * sc + mn
    df = 2 * a; s = np.sqrt(b * (1 + nu) / (nu * a)) * sc
    ale = b / (a - 1); epi = b / (nu * (a - 1)); R = epi / (ale + epi)
    lp = tdist.logpdf(Y, df, loc=P, scale=s)
    qs = np.linspace(0.01, 0.99, 99); pin = []
    for q in qs:
        Q = P + tdist.ppf(q, df) * s; dd = Y - Q; pin.append(np.mean(np.maximum(q * dd, (q - 1) * dd)))
    obs = np.array([np.mean(np.abs(Y - P) <= tdist.ppf((1 + c) / 2, df) * s) for c in LEVELS])
    return dict(MAE=np.mean(np.abs(Y - P)), NLL=-np.mean(lp), CRPS=2 * np.mean(pin), ECE=np.mean(np.abs(LEVELS - obs)),
                PICP90=obs[-2], R_epi=np.mean(R), R_epi_med=np.median(R), nu_med=np.median(nu), alpha_med=np.median(a),
                sig_ale=np.mean(np.sqrt(ale)) * sc, sig_epi=np.mean(np.sqrt(epi)) * sc)

if __name__ == "__main__":
    d = prep_daystrat("GEF"); sdl = d["sdl"]
    runs = [(lam, s) for lam in [0.0, 1e-3, 1e-2, 1e-1, 1.0] for s in [1, 2]] + [(0.0, 3), (1e-2, 3)]
    res = []
    for lam, s in runs:
        path = f"{OUTA}/der_lam{lam}_s{s}.pth"
        MM.setup_seed(s); m = DER(d["sl"], d["nv"])
        if not os.path.exists(path): train_der(m, d, s, lam, path)
        else: m.load_state_dict(torch.load(path, weights_only=True))
        te = scores(d["tel"], *pred(m, d["te"]), sdl); ca = scores(d["call"], *pred(m, d["cal"]), sdl)
        r = dict(lam=lam, seed=s, cal_NLL=ca["NLL"], cal_CRPS=ca["CRPS"], **te); res.append(r)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
    # exact reparameterisation along the likelihood-flat direction (lambda=0.01, seed 1)
    m = DER(d["sl"], d["nv"]); m.load_state_dict(torch.load(f"{OUTA}/der_lam0.01_s1.pth", weights_only=True))
    g, nu, a, b = pred(m, d["te"]); k = b * (1 + nu) / nu
    for c in [0.1, 0.3, 1.0, 3.0, 10.0]:
        nu2 = nu * c; b2 = k * nu2 / (1 + nu2); sc_ = scores(d["tel"], g, nu2, a, b2, sdl)
        r = dict(lam="reparam", seed=c, **sc_); res.append(r)
        print("reparam nu x", c, " ".join(f"{k_}={v:.4g}" for k_, v in sc_.items()), flush=True)
    json.dump(res, open(f"{OUT}/der_evidential.json", "w"), indent=1, default=float)
    print("DONE")
