"""Appendix F: quantile-regression reserve policy as an external reference for D2.
Schedule S = Q_QR(tau*) directly (no calibration multiplier); same cost function as sec62.py.
Also recomputes the ensemble total-uncertainty policy from the saved K=10 predictions as a check,
and a paired moving-block bootstrap CI for (QR cost - total cost)/total cost."""
from common import *
import quantile_regression_baseline as QR
LOG=open(f"{OUT}/qr_reserve_log.txt","w")
def log(x): print(x,flush=True); LOG.write(x+"\n"); LOG.flush()
B=1000; BLOCK=7; rng=np.random.RandomState(0)
def loaders(d,bs=256):
    mk=lambda X,y,sh: torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.float32).unsqueeze(1)),batch_size=bs,shuffle=sh)
    return mk(d["tr"],d["trl"],True), mk(d["es"],d["esl"],False)
def mbb(nd):
    out=[]
    while len(out)<nd:
        s=rng.randint(0,nd-BLOCK+1); out+=list(range(s,s+BLOCK))
    return np.array(out[:nd])
for name in ["GEF","COVID"]:
    d=prep_daystrat(name); sc,mn=d["sdl"].scale_[0],d["sdl"].mean_[0]; div=1000 if name=="COVID" else 1
    path=f"{OUT}/{name}_qr_seed1.pth"; MM.setup_seed(1); tl,vl=loaders(d); MM.setup_seed(1); qr=QR.QuantileNet(d["sl"],d["nv"])
    if os.path.exists(path): qr.load_state_dict(torch.load(path,weights_only=True))
    else: QR.train_qr(qr,tl,vl,epochs=50,patience=12,save=path)
    qr.eval()
    with torch.no_grad(): Q=np.sort(qr(torch.tensor(d["te"],dtype=torch.float32)).numpy(),1)*sc+mn
    P=np.load(f"{OUT}/{name}_ensK10_preds.npz"); mu,ale,epi,muc,alec,epic,y,yc=[P[k] for k in ["mu","ale","epi","muc","alec","epic","y","yc"]]
    epi=epi*np.sqrt(10/9); epic=epic*np.sqrt(10/9)
    Yt=y*sc+mn; Yc=yc*sc+mn; Mt=mu*sc+mn; Mc=muc*sc+mn; nd=len(Yt)//24; hours=lambda di: (di[:,None]*24+np.arange(24)[None,:]).ravel()
    log(f"\n{name}: QR reserve policy (S = QR tau*-quantile, no calibration) vs ensemble total-uncertainty policy; cost per hour in MW")
    log(f"  QR test MAE (median) = {np.mean(np.abs(Yt-Q[:,49]))/div:.2f} MW ; ensemble MAE = {np.mean(np.abs(Yt-Mt))/div:.2f} MW")
    for ratio in [2,5,10,20]:
        tau=ratio/(ratio+1.0); cost=lambda Y,S: ratio*np.maximum(Y-S,0)+np.maximum(S-Y,0)
        # QR quantile at tau* by linear interpolation on the 0.01..0.99 grid
        g=QR.QGRID; i=np.searchsorted(g,tau)-1; w=(tau-g[i])/(g[i+1]-g[i]); Sq=(1-w)*Q[:,i]+w*Q[:,i+1]
        cq=cost(Yt,Sq)
        sgt=np.sqrt(ale**2+epi**2); sgc=np.sqrt(alec**2+epic**2); q=np.quantile((yc-muc)/sgc,tau); ct=cost(Yt,Mt+q*sgt*sc)
        cd=cost(Yt,Mt)
        rel=100*(cq.mean()-ct.mean())/ct.mean()
        vals=[]
        for _ in range(B):
            h=hours(mbb(nd)); vals.append(100*(cq[h].mean()-ct[h].mean())/ct[h].mean())
        lo,hi=np.percentile(vals,[2.5,97.5])
        log(f"  {ratio:2d}:1  deterministic={cd.mean()/div:8.2f}  ensemble total={ct.mean()/div:8.2f}  QR tau*-quantile={cq.mean()/div:8.2f}   QR vs total = {rel:+.2f}% [95% CI {lo:+.2f}, {hi:+.2f}]")
log("DONE")
