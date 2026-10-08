"""
Heteroscedastic iTransformer forecaster and moment-based uncertainty decomposition
(manuscript Section 3.2, Eqs. (10)-(16)).

- Each parameter realization theta^(i) ~ Q (an MC-dropout mask or a deep-ensemble member) returns
  a conditional mean mu^(i) and a log aleatoric variance log sigma^2_ale^(i).
- The network is trained by the heteroscedastic Gaussian NLL only (Eq. (11)).
- After training, over S realizations:
      sigma^2_ale = mean_i sigma^2_ale^(i)          (E_Q[Var(y | x, theta)])
      sigma^2_epi = Var_i mu^(i)                    (Var_Q[E(y | x, theta)])
  `predict` returns the population variance (ddof = 0); the experiment scripts apply the
  Bessel correction S/(S-1) used in the paper.
- The epistemic term is a derived statistic, not a learned output. Its scale still depends on the
  mechanism that defines Q (dropout rate, ensemble construction, training settings); see Sections 4.4
  and 7.2 of the manuscript.

The latent diffusion module of an earlier version is kept only so that old checkpoints load; it is
disabled (use_diffusion=False, lambda_diff=0) in every experiment reported in the paper.
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import properscoring as ps
import diffmod as M   # reuse iTransformerEncoder, MLPDiffusion, DiffusionSchedule, TimeSeriesDataset

device = torch.device("cpu"); M.device = device


class HeteroHead(nn.Module):
    """Outputs the mean and the log aleatoric variance; there is no epistemic output."""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.logvar_head.bias)  # var ~ 1 in normalized space at init

    def forward(self, h):
        f = self.shared(h)
        mu = self.mu_head(f).squeeze(-1)
        logvar = torch.clamp(self.logvar_head(f).squeeze(-1), min=-7.0, max=3.0)
        return mu, logvar


class IdentDiffLoad(nn.Module):
    def __init__(self, seq_len, n_var, d_model=64, n_heads=4, n_layers=2,
                 num_diff_steps=5, dropout=0.1, use_diffusion=True):
        super().__init__()
        self.encoder = M.iTransformerEncoder(seq_len, n_var, d_model, n_heads, n_layers, dropout)
        self.num_diff_steps = num_diff_steps
        self.use_diffusion = use_diffusion
        self.diffusion_net = M.MLPDiffusion(num_diff_steps, d_model)
        self.head = HeteroHead(d_model, dropout)
        self.diff_schedule = None

    def set_schedule(self, s): self.diff_schedule = s

    def forward(self, x):
        h = self.encoder(x)
        diff_loss = torch.tensor(0.0, device=x.device)
        if self.use_diffusion and self.diff_schedule is not None:
            hn, _ = self.diff_schedule.q_sample(h, self.num_diff_steps - 1)
            diff_loss = self.diff_schedule.compute_loss(self.diffusion_net, h)
            h = self.diff_schedule.p_sample(self.diffusion_net, hn, self.num_diff_steps - 1)
        mu, logvar = self.head(h)
        return mu, logvar, diff_loss


def enable_mc_dropout(model):
    """Turn ON dropout (and only dropout) for inference-time MC sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def hetero_nll(y, mu, logvar):
    return 0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar)).mean()


def train(model, tr_loader, vd_loader, schedule, epochs=50, patience=12, lr=5e-4,
          lambda_diff=1.0, save="best.pth"):
    model.set_schedule(schedule); model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5, min_lr=1e-6)
    best = float('inf'); pc = 0
    for ep in range(epochs):
        model.train()
        for x, y in tr_loader:
            x = x.to(device); y = y.squeeze().to(device)
            mu, logvar, dl = model(x)
            loss = hetero_nll(y, mu, logvar) + lambda_diff * dl
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vr = []
        with torch.no_grad():
            for x, y in vd_loader:
                x = x.to(device); y = y.squeeze().to(device)
                mu, _, _ = model(x)
                vr.append(torch.sqrt(((y - mu) ** 2).mean()).item())
        vr = float(np.mean(vr)); sch.step(vr)
        if vr < best:
            best = vr; pc = 0
            torch.save(model.state_dict(), save)
        else:
            pc += 1
            if pc >= patience: break
    model.load_state_dict(torch.load(save, weights_only=True))
    return best


@torch.no_grad()
def predict(model, X, schedule, n_mc=40, mc_dropout=True):
    model.set_schedule(schedule); model.to(device)
    if mc_dropout: enable_mc_dropout(model)
    else: model.eval()
    mus, varales = [], []
    for _ in range(n_mc):
        mu, logvar, _ = model(X.to(device))
        mus.append(mu.cpu().numpy()); varales.append(np.exp(logvar.cpu().numpy()))
    mus = np.stack(mus, 0); varales = np.stack(varales, 0)
    mu_bar = mus.mean(0)
    var_ale = varales.mean(0)          # E_theta[Var]  -> aleatoric
    var_epi = mus.var(0)               # Var_theta[E]  -> epistemic (population variance; Bessel applied in scripts)
    var_tot = var_ale + var_epi
    return dict(mu=mu_bar, ale=np.sqrt(var_ale), epi=np.sqrt(var_epi),
                tot=np.sqrt(var_tot), R_epi=float(np.mean(var_epi / (var_tot + 1e-12))))


def prep(frac=1.0, seed=0):
    d = os.environ.get("UQ_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "GEFCom2014"))
    tr = np.load(f"{d}/train_data.npy"); trl = np.load(f"{d}/train_label.npy")
    vd = np.load(f"{d}/val_data.npy");  vdl = np.load(f"{d}/val_label.npy")
    te = np.load(f"{d}/test_data.npy"); tel = np.load(f"{d}/test_label.npy")
    if frac < 1.0:
        n = int(len(tr) * frac); tr = tr[-n:]; trl = trl[-n:]   # keep most recent frac
    sl, nv = tr.shape[1], tr.shape[2]
    sdd = StandardScaler().fit(tr.reshape(-1, nv)); sdl = StandardScaler().fit(trl.reshape(-1, 1))
    f = lambda a: sdd.transform(a.reshape(-1, nv)).reshape(-1, sl, nv)
    g = lambda a: sdl.transform(a.reshape(-1, 1)).flatten()
    return (f(tr), g(trl), f(vd), g(vdl), f(te), g(tel), sl, nv, sdl)


def metrics(y, mu, sig, scale, mean):
    Y = y * scale + mean; P = mu * scale + mean; S = sig * scale
    mae = np.mean(np.abs(Y - P)); mape = np.mean(np.abs((Y - P) / (np.abs(Y) + 1e-8))) * 100
    crps = np.mean(ps.crps_gaussian(Y, P, S))
    levels = np.linspace(0.5, 0.95, 10); obs = []
    for c in levels:
        z = norm.ppf((1 + c) / 2); obs.append(np.mean((Y >= P - z * S) & (Y <= P + z * S)))
    ece = np.mean(np.abs(levels - np.array(obs)))
    z90 = norm.ppf(0.95); picp90 = np.mean((Y >= P - z90 * S) & (Y <= P + z90 * S))
    return dict(MAE=mae, MAPE=mape, CRPS=crps, ECE=ece, PICP90=picp90)
