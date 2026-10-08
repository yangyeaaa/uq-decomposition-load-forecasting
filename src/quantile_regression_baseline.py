import os
import numpy as np, torch, torch.nn as nn
import diffmod as MM
import decomposition as FX
from scipy.stats import norm

device = torch.device("cpu"); MM.device = device; FX.device = device

# quantile grid (dense) for training/eval
QGRID = np.round(np.arange(0.01, 1.00, 0.01), 2)   # 0.01 .. 0.99
QT = torch.tensor(QGRID, dtype=torch.float32)

class QuantileNet(nn.Module):
    def __init__(self, seq_len, n_var, d_model=64, n_heads=4, n_layers=2, dropout=0.1, nq=len(QGRID)):
        super().__init__()
        self.encoder = MM.iTransformerEncoder(seq_len, n_var, d_model, n_heads, n_layers, dropout)
        self.head = nn.Linear(d_model, nq)
    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)            # (B, nq)

def pinball_loss(pred, y):
    # pred (B,nq), y (B,)
    e = y.unsqueeze(1) - pred          # (B,nq)
    q = QT.to(pred.device).unsqueeze(0)
    return torch.maximum(q * e, (q - 1) * e).mean()

def train_qr(model, tl, vl, epochs=50, patience=12, lr=5e-4, save="qr.pth"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5, min_lr=1e-6)
    best = float('inf'); pc = 0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x = x.to(device); y = y.squeeze().to(device)
            loss = pinball_loss(model(x), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vr = []
        with torch.no_grad():
            for x, y in vl:
                x = x.to(device); y = y.squeeze().to(device)
                vr.append(pinball_loss(model(x), y).item())
        vr = float(np.mean(vr)); sch.step(vr)
        if vr < best: best = vr; pc = 0; torch.save(model.state_dict(), save)
        else:
            pc += 1
            if pc >= patience: break
    model.load_state_dict(torch.load(save, weights_only=True))
    return best

def interp_quantile(Q, tau):
    # Q: (N,nq) sorted along axis1 ; return quantile at level tau via per-row interp over QGRID
    return np.array([np.interp(tau, QGRID, Q[i]) for i in range(Q.shape[0])])

def evaluate(model, ten, teln, scale, mean):
    model.eval()
    with torch.no_grad():
        Q = model(torch.tensor(ten, dtype=torch.float32).to(device)).cpu().numpy()
    Q = np.sort(Q, axis=1)                       # enforce monotone quantiles
    Q = Q * scale + mean                         # to original units
    y = teln * scale + mean
    med = Q[:, np.argmin(np.abs(QGRID - 0.50))]
    mae = np.mean(np.abs(y - med))
    mape = np.mean(np.abs((y - med) / (np.abs(y) + 1e-8))) * 100
    # CRPS via pinball integral: CRPS = 2 * mean_tau PB_tau
    e = y[:, None] - Q
    pb = np.maximum(QGRID[None, :] * e, (QGRID[None, :] - 1) * e)
    crps = 2.0 * pb.mean()
    # coverage / ECE over central intervals c in [0.5,0.95]
    levels = np.linspace(0.5, 0.95, 10); obs = []
    for c in levels:
        lo = interp_quantile(Q, (1 - c) / 2); hi = interp_quantile(Q, (1 + c) / 2)
        obs.append(np.mean((y >= lo) & (y <= hi)))
    ece = np.mean(np.abs(levels - np.array(obs)))
    lo90 = interp_quantile(Q, 0.05); hi90 = interp_quantile(Q, 0.95)
    picp90 = np.mean((y >= lo90) & (y <= hi90))
    return dict(MAE=mae, MAPE=mape, CRPS=crps, ECE=ece, PICP90=picp90)

if __name__ == "__main__":
    MM.setup_seed(1)
    trn,trln,vdn,vdln,ten,teln,sl,nv,sdl = FX.prep(frac=1.0, seed=1)
    tl = torch.utils.data.DataLoader(MM.TimeSeriesDataset(trn,trln), batch_size=256, shuffle=True)
    vl = torch.utils.data.DataLoader(MM.TimeSeriesDataset(vdn,vdln), batch_size=256, shuffle=False)
    model = QuantileNet(sl, nv, 64, 4, 2, 0.1)
    train_qr(model, tl, vl, epochs=50, patience=12, save=os.path.join(os.environ.get("UQ_OUT","results/out"),"qr_seed1.pth"))
    m = evaluate(model, ten, teln, sdl.scale_[0], sdl.mean_[0])
    line = (f"Quantile Regression: MAE={m['MAE']:.2f} MAPE={m['MAPE']:.3f} "
            f"CRPS={m['CRPS']:.2f} ECE={m['ECE']:.4f} PICP90={m['PICP90']:.4f}")
    print(line)
    open(os.path.join(os.environ.get("UQ_OUT","results/out"),"qr_result.txt"),"w").write(line+"\n")
