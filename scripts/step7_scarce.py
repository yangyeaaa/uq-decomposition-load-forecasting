"""Step 7: data-scarce regime. Train on the first M days of the training partition, early-stop /
calibrate on the following 14 days, test on the fixed test partition. Ensemble K=5 per M."""
from common import *
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import json
LOG=open(f"{OUT}/step7_log.txt","w")
def log(s): print(s,flush=True); LOG.write(s+"\n"); LOG.flush()
M_LIST=[30,60,90,180,None]

def make_split(name, M, seed=0):
    r=load_raw(name); tr,trl=r["train"]; vd,vdl=r["val"]; te,tel=r["test"]; sl,nv=tr.shape[1],tr.shape[2]
    if M is None: Xtr,ytr=tr,trl; Xh,yh=vd[:14*24],vdl[:14*24]
    else: Xtr,ytr=tr[:M*24],trl[:M*24]; Xh,yh=tr[M*24:(M+14)*24],trl[M*24:(M+14)*24]
    sdd=StandardScaler().fit(Xtr.reshape(-1,nv)); sdl=StandardScaler().fit(ytr.reshape(-1,1))
    f=lambda a: sdd.transform(a.reshape(-1,nv)).reshape(-1,sl,nv); g=lambda a: sdl.transform(a.reshape(-1,1)).flatten()
    rng=np.random.RandomState(seed); ies,ical=[],[]
    for d0 in range(0,len(Xh),24):
        idx=np.arange(d0,min(d0+24,len(Xh))); rng.shuffle(idx); h=len(idx)//2; ies+=list(idx[:h]); ical+=list(idx[h:])
    ies,ical=np.sort(ies),np.sort(ical)
    return dict(tr=f(Xtr),trl=g(ytr),es=f(Xh[ies]),esl=g(yh[ies]),cal=f(Xh[ical]),call=g(yh[ical]),te=f(te),tel=g(tel),sl=sl,nv=nv,sdl=sdl)

def ensemble(name,M,d):
    mus_t,vs_t,mus_c,vs_c=[],[],[],[]; ttrain=0
    for k in range(5):
        m=make_model(d,0.1,seed=10+k); path=f"{OUT}/{name}_M{M}_ens{k}.pth"
        if os.path.exists(path): m.load_state_dict(torch.load(path,weights_only=True))
        else: ttrain+=train(m,d,10+k,path)
        rt=predict(m,d["te"],1,False); rc=predict(m,d["cal"],1,False)
        mus_t.append(rt["mu"]); vs_t.append(rt["ale"]**2); mus_c.append(rc["mu"]); vs_c.append(rc["ale"]**2)
    A=lambda L: np.stack(L)
    return (A(mus_t).mean(0),np.sqrt(A(vs_t).mean(0)),np.sqrt(A(mus_t).var(0))),(A(mus_c).mean(0),np.sqrt(A(vs_c).mean(0)),np.sqrt(A(mus_c).var(0))),ttrain

def reserve_gain(y_c,mu_c,ale_c,epi_c,y_t,mu_t,ale_t,epi_t,ratio,sc,mn):
    tau=ratio/(ratio+1.0); Yc=y_c*sc+mn; Yt=y_t*sc+mn; Mc=mu_c*sc+mn; Mt=mu_t*sc+mn
    cost=lambda Y,S: np.mean(ratio*np.maximum(Y-S,0)+np.maximum(S-Y,0))
    res={}
    for c in [0,0.5,1,1.5,2,3,4,6]:
        sg=np.sqrt(ale_c**2+(c*epi_c)**2); q=np.quantile((y_c-mu_c)/sg,tau)
        res[c]=(cost(Yc,Mc+q*sg*sc), cost(Yt,Mt+q*np.sqrt(ale_t**2+(c*epi_t)**2)*sc))
    cstar=min(res,key=lambda c:res[c][0]); return res[0][1],res[1][1],res[cstar][1],cstar

def run(name):
    log(f"\n{'='*70}\n{name}\n{'='*70}")
    hdr=f"{'M(days)':>8s}{'MAE':>9s}{'CRPS':>9s}{'R_epi':>7s}{'sig_epi':>9s}{'sig_ale':>9s}{'PICP90 ale-only':>16s}{'PICP90 total':>13s}{'AUC epi':>9s}{'AUC tot':>9s}{'AUC ale':>9s}{'MAE|flag10% epi':>16s}{'tot':>8s}{'reserve10:1 gain c*':>20s}{'train s':>9s}"
    log(hdr); rows=[]
    for M in M_LIST:
        d=make_split(name,M); sc,mn=d["sdl"].scale_[0],d["sdl"].mean_[0]
        (mu_t,ale_t,epi_t),(mu_c,ale_c,epi_c),tt=ensemble(name,M,d)
        y_t=d["tel"]; tot_t=np.sqrt(ale_t**2+epi_t**2); err=np.abs(y_t-mu_t); m=full_metrics(y_t,mu_t,tot_t,d["sdl"])
        z=norm.ppf(0.95); picp_ale=np.mean(err<=z*ale_t); picp_tot=np.mean(err<=z*tot_t)
        top10=err>=np.quantile(err,0.9)
        auc=lambda s: roc_auc_score(top10,s)
        rc=lambda s: err[s<np.quantile(s,0.9)].mean()*sc
        c0,c1,cs,cstar=reserve_gain(d["call"],mu_c,ale_c,epi_c,y_t,mu_t,ale_t,epi_t,10,sc,mn)
        R=np.mean(epi_t**2/tot_t**2)
        row=dict(M=M if M else 359,MAE=m["MAE"],CRPS=m["CRPS"],R_epi=R,epi=epi_t.mean()*sc,ale=ale_t.mean()*sc,picp_ale=picp_ale,picp_tot=picp_tot,
                 auc_epi=auc(epi_t),auc_tot=auc(tot_t),auc_ale=auc(ale_t),rc_epi=rc(epi_t),rc_tot=rc(tot_t),gain=100*(c1-cs)/c1,cstar=cstar,train=tt,
                 rho_epi_err=spearmanr(epi_t,err)[0],rho_ale_err=spearmanr(ale_t,err)[0])
        rows.append(row)
        log(f"{row['M']:>8d}{m['MAE']:9.1f}{m['CRPS']:9.1f}{R:7.3f}{row['epi']:9.1f}{row['ale']:9.1f}{picp_ale:16.3f}{picp_tot:13.3f}{row['auc_epi']:9.3f}{row['auc_tot']:9.3f}{row['auc_ale']:9.3f}{row['rc_epi']:16.1f}{row['rc_tot']:8.1f}{row['gain']:14.2f}% (c={cstar}){tt:9.0f}")
        np.savez(f"{OUT}/{name}_M{M}_preds.npz",mu=mu_t,ale=ale_t,epi=epi_t,y=y_t,scale=sc,mean=mn)
    log("rho(sigma_epi,|e|) by M: "+", ".join(f"{r['M']}:{r['rho_epi_err']:.3f}" for r in rows)+"   rho(sigma_ale,|e|): "+", ".join(f"{r['M']}:{r['rho_ale_err']:.3f}" for r in rows))
    json.dump(rows,open(f"{OUT}/{name}_step7.json","w"),indent=1,default=float)

if __name__=="__main__":
    for name in ["GEF","COVID"]: run(name)
    log("DONE")
