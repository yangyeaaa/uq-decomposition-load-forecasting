"""Shared data pipeline, training loop, and scoring helpers used by every experiment script."""
import sys, os, time, numpy as np, torch
torch.set_num_threads(int(os.environ.get("UQ_THREADS", "1")))   # single thread: reproducible timings and results
ROOT = os.environ.get("UQ_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.insert(0, f"{ROOT}/src")
import diffmod as MM, decomposition as FX
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import properscoring as ps
DEV = torch.device("cpu"); MM.device = DEV; FX.device = DEV
DATA = {"GEF": f"{ROOT}/data/GEFCom2014", "COVID": f"{ROOT}/data/postCOVID"}
SPLIT = os.environ.get("SPLIT","hour")
OUT = os.environ.get("UQ_OUT", f"{ROOT}/results/out_alt" if SPLIT=="alt" else f"{ROOT}/results/out"); os.makedirs(OUT, exist_ok=True)

def load_raw(name):
    D = DATA[name]
    return {s: (np.load(f"{D}/{s}_data.npy"), np.load(f"{D}/{s}_label.npy")) for s in ["train","val","test"]}

def prep(name, val_split=None, seed=0):
    """val_split=None -> original (val used for both). val_split=(frac_es) -> split val
    chronologically-interleaved into early-stopping and calibration halves."""
    r = load_raw(name)
    tr,trl = r["train"]; vd,vdl = r["val"]; te,tel = r["test"]
    sl,nv = tr.shape[1],tr.shape[2]
    sdd = StandardScaler().fit(tr.reshape(-1,nv)); sdl = StandardScaler().fit(trl.reshape(-1,1))
    f = lambda a: sdd.transform(a.reshape(-1,nv)).reshape(-1,sl,nv)
    g = lambda a: sdl.transform(a.reshape(-1,1)).flatten()
    out = dict(tr=f(tr), trl=g(trl), te=f(te), tel=g(tel), sl=sl, nv=nv, sdl=sdl)
    if val_split is None:
        out["es"]=f(vd); out["esl"]=g(vdl); out["cal"]=f(vd); out["call"]=g(vdl)
    else:
        # random disjoint split of the validation partition (fixed seed), so both halves cover all seasons
        rng = np.random.RandomState(seed); idx = rng.permutation(len(vd))
        n_es = int(len(vd)*val_split); ies, ical = np.sort(idx[:n_es]), np.sort(idx[n_es:])
        out["es"]=f(vd[ies]); out["esl"]=g(vdl[ies]); out["cal"]=f(vd[ical]); out["call"]=g(vdl[ical])
    return out

def make_model(d, dropout=0.1, seed=None):
    if seed is not None: MM.setup_seed(seed)
    return FX.IdentDiffLoad(d["sl"], d["nv"], 64, 4, 2, 5, dropout=dropout, use_diffusion=False)

def loaders(d):
    tl = torch.utils.data.DataLoader(MM.TimeSeriesDataset(d["tr"],d["trl"]),batch_size=256,shuffle=True)
    vl = torch.utils.data.DataLoader(MM.TimeSeriesDataset(d["es"],d["esl"]),batch_size=256,shuffle=False)
    return tl, vl

def train(model, d, seed, save, epochs=50, patience=12):
    MM.setup_seed(seed); tl, vl = loaders(d)
    sched = MM.DiffusionSchedule(5, DEV)
    t0 = time.time(); FX.train(model, tl, vl, sched, epochs=epochs, patience=patience, lambda_diff=0.0, save=save)
    return time.time()-t0

def predict(model, X, n_mc=40, mc=True):
    return FX.predict(model, torch.tensor(X,dtype=torch.float32), MM.DiffusionSchedule(5,DEV), n_mc=n_mc, mc_dropout=mc)

LEVELS = np.linspace(0.5,0.95,10)
def full_metrics(y, mu, sig, sdl):
    """Point + proper scores + calibration, all in original units."""
    sc,mn = sdl.scale_[0], sdl.mean_[0]
    Y=y*sc+mn; P=mu*sc+mn; S=sig*sc
    e=Y-P
    m=dict(MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e**2)), MAPE=100*np.mean(np.abs(e/Y)))
    m["CRPS"]=np.mean(ps.crps_gaussian(Y,P,S))
    m["NLL"]=np.mean(0.5*np.log(2*np.pi*S**2)+0.5*(e/S)**2)
    qs=np.arange(0.05,0.96,0.05); pin=[]
    for q in qs:
        Q=P+norm.ppf(q)*S; d=Y-Q; pin.append(np.mean(np.maximum(q*d,(q-1)*d)))
    m["Pinball"]=np.mean(pin)
    z=norm.ppf(0.95); lo,hi=P-z*S,P+z*S; a=0.1
    m["IS90"]=np.mean((hi-lo)+2/a*(lo-Y)*(Y<lo)+2/a*(Y-hi)*(Y>hi))
    obs=np.array([np.mean(np.abs(e)<=norm.ppf((1+c)/2)*S) for c in LEVELS])
    m["ECE"]=np.mean(np.abs(LEVELS-obs)); m["PICP90"]=obs[-2]  # levels[-2]=0.90
    return m

def conformal_q(ycal, mucal, sigcal, level):
    s=np.abs(ycal-mucal)/sigcal; n=len(s); k=min(n,int(np.ceil((n+1)*level)))
    return np.sort(s)[k-1]

def conformal_metrics(ycal,mucal,sigcal, y,mu,sig, sdl):
    """Per-level conformal intervals; returns ECE, PICP90, IS90 (conformal) in original units."""
    sc,mn=sdl.scale_[0],sdl.mean_[0]; Y=y*sc+mn; P=mu*sc+mn; S=sig*sc
    obs=[]
    for c in LEVELS:
        q=conformal_q(ycal,mucal,sigcal,c); obs.append(np.mean(np.abs(Y-P)<=q*S))
    obs=np.array(obs); q90=conformal_q(ycal,mucal,sigcal,0.90); lo,hi=P-q90*S,P+q90*S; a=0.1
    return dict(ECE_conf=np.mean(np.abs(LEVELS-obs)), PICP90_conf=obs[-2],
                IS90_conf=np.mean((hi-lo)+2/a*(lo-Y)*(Y<lo)+2/a*(Y-hi)*(Y>hi)), q90=q90)

def fmt(m): return " ".join(f"{k}={v:.4g}" for k,v in m.items())


def prep_hoursplit(name, seed=0):
    """Split the validation partition 50/50 within each day (12 hours to early stopping,
    12 to calibration), so both halves cover every day and season."""
    r = load_raw(name); tr,trl = r["train"]; vd,vdl = r["val"]; te,tel = r["test"]
    sl,nv = tr.shape[1],tr.shape[2]
    sdd = StandardScaler().fit(tr.reshape(-1,nv)); sdl = StandardScaler().fit(trl.reshape(-1,1))
    f = lambda a: sdd.transform(a.reshape(-1,nv)).reshape(-1,sl,nv)
    g = lambda a: sdl.transform(a.reshape(-1,1)).flatten()
    rng = np.random.RandomState(seed); n = len(vd); ies, ical = [], []
    for d0 in range(0, n, 24):
        idx = np.arange(d0, min(d0+24, n)); rng.shuffle(idx)
        h = len(idx)//2; ies += list(idx[:h]); ical += list(idx[h:])
    ies, ical = np.sort(ies), np.sort(ical)
    return dict(tr=f(tr), trl=g(trl), es=f(vd[ies]), esl=g(vdl[ies]), cal=f(vd[ical]), call=g(vdl[ical]),
                te=f(te), tel=g(tel), sl=sl, nv=nv, sdl=sdl, n_es=len(ies), n_cal=len(ical))


def prep_altday(name, seed=0):
    """Alternating complete target days: even days -> early stopping, odd days -> calibration."""
    r=load_raw(name); tr,trl=r["train"]; vd,vdl=r["val"]; te,tel=r["test"]; sl,nv=tr.shape[1],tr.shape[2]
    sdd=StandardScaler().fit(tr.reshape(-1,nv)); sdl=StandardScaler().fit(trl.reshape(-1,1))
    f=lambda a: sdd.transform(a.reshape(-1,nv)).reshape(-1,sl,nv); g=lambda a: sdl.transform(a.reshape(-1,1)).flatten()
    day=np.arange(len(vd))//24; ies=np.where(day%2==0)[0]; ical=np.where(day%2==1)[0]
    return dict(tr=f(tr),trl=g(trl),es=f(vd[ies]),esl=g(vdl[ies]),cal=f(vd[ical]),call=g(vdl[ical]),te=f(te),tel=g(tel),sl=sl,nv=nv,sdl=sdl,n_es=len(ies),n_cal=len(ical))

def prep_daystrat(name, seed=0):
    return prep_altday(name, seed) if SPLIT=="alt" else prep_hoursplit(name, seed)
