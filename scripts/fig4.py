from common import *
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=prep_daystrat("GEF"); sc,mn=d["sdl"].scale_[0],d["sdl"].mean_[0]; y=d["tel"]; Y=y*sc+mn; Xte=d["te"]
def ens(prefix,K,X=Xte,dd=d):
    mus,vs=[],[]
    for k in range(K):
        m=make_model(dd,0.1); m.load_state_dict(torch.load(f"{OUT}/{prefix}{k}.pth",weights_only=True)); r=predict(m,X,1,False); mus.append(r["mu"]); vs.append(r["ale"]**2)
    mus,vs=np.stack(mus),np.stack(vs); return mus.mean(0),np.sqrt(vs.mean(0)),np.sqrt(mus.var(0,ddof=1))
def decile(e):
    edges=np.percentile(Y,np.linspace(0,100,11)); out=[]
    for i in range(10):
        msk=(Y>=edges[i])&((Y<edges[i+1]) if i<9 else (Y<=edges[i+1])); out.append(e[msk].mean()*sc)
    return np.array(out)
# (a) decile profiles
prof={}
for p in [0.05,0.10,0.20,0.30]:
    m=make_model(d,p); m.load_state_dict(torch.load(f"{OUT}/GEF_p{int(p*100)}_seed1.pth",weights_only=True)); prof[f"MC dropout, $p={p:.2f}$"]=decile(predict(m,Xte,40,True)["epi"]*np.sqrt(40/39))
mu10,ale10,epi10=ens("GEF_ens",10); prof["Ensemble, $K=10$"]=decile(epi10)
# (b) held-out band: K=5 band-holdout members vs K=5 full-data members (same seeds 10-14)
zT=Xte[:,-1,2]; inb=zT>1.0
mub,aleb,epib=ens("band_m",5); muf,alef,epif=ens("GEF_ens",5)
errb=np.abs(y-mub)*sc; errf=np.abs(y-muf)*sc
bvals=dict(epi_hold=(epib[~inb].mean()*sc,epib[inb].mean()*sc),epi_full=(epif[~inb].mean()*sc,epif[inb].mean()*sc),err_hold=(errb[~inb].mean(),errb[inb].mean()),err_full=(errf[~inb].mean(),errf[inb].mean()))
open(f"{OUT}/fig4_values.txt","w").write(str({k:[round(v,1) for v in vv] for k,vv in bvals.items()})+"\n"+"\n".join(f"{k}: {np.round(v,1).tolist()}" for k,v in prof.items()))
# (c) temperature shift values (from tables)
dl=[0,0.5,1.0,1.5,2.0]; e_ens=[33.3,38.2,48.7,63.1,81.0]; a_ens=[135.0,143.9,156.6,172.5,190.7]; r_ens=[0.061,0.069,0.091,0.117,0.147]
e_mcd=[39.0,39.7,40.8,42.5,45.0]; a_mcd=[155.8,162.9,172.1,183.5,198.6]; r_mcd=[0.067,0.063,0.060,0.057,0.055]; frac=[0.00,0.02,0.15,0.35,0.53]
B,G,O="#3b6aa5","#3a8f6e","#d9641e"
fig,axes=plt.subplots(1,3,figsize=(15,4.4))
ax=axes[0]; blues=["#c6d4ea","#8fabd3","#5a82bb","#2f5a9a"]
for (lab,v),c in zip([(k,v) for k,v in prof.items() if k.startswith("MC")],blues): ax.plot(range(1,11),v,"o-",color=c,lw=1.4,ms=4,label=lab)
ax.plot(range(1,11),prof["Ensemble, $K=10$"],"s-",color=G,lw=2.2,ms=5,label="Ensemble, $K=10$")
ax.set_xlabel("Decile of realized load"); ax.set_ylabel(r"Mean $\hat\sigma_{\rm epi}$ (MW)"); ax.set_title("(a) Load-decile profile",loc="left",fontweight="bold"); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8.5); ax.set_xticks(range(1,11))
ax=axes[1]; w=0.35; x=np.arange(2)
ax.bar(x-w/2,bvals["epi_full"],w,color=G,alpha=0.45,label=r"$\hat\sigma_{\rm epi}$, band in training")
ax.bar(x+w/2,bvals["epi_hold"],w,color=G,label=r"$\hat\sigma_{\rm epi}$, band withheld")
ax2=ax.twinx(); ax2.plot(x-w/2,bvals["err_full"],"o",color="#444",ms=7,mfc="white",label="MAE, band in training"); ax2.plot(x+w/2,bvals["err_hold"],"o",color="#444",ms=7,label="MAE, band withheld")
ax.set_xticks(x); ax.set_xticklabels([f"outside band\n(n={int((~inb).sum())})",f"inside band $z_T>1$\n(n={int(inb.sum())})"]); ax.set_ylabel(r"Mean $\hat\sigma_{\rm epi}$ (MW)",color=G); ax2.set_ylabel("MAE (MW)",color="#444")
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,frameon=False,fontsize=8,loc="upper left"); ax.set_title("(b) Held-out temperature band ($K=5$)",loc="left",fontweight="bold"); ax.set_ylim(0,max(bvals["epi_hold"])*1.9); ax2.set_ylim(0,max(bvals["err_hold"])*1.35)
ax=axes[2]
ax.plot(dl,e_ens,"s-",color=G,lw=2,label=r"Ensemble $\hat\sigma_{\rm epi}$"); ax.plot(dl,a_ens,"s--",color=G,lw=1.4,alpha=.7,label=r"Ensemble $\hat\sigma_{\rm ale}$")
ax.plot(dl,e_mcd,"o-",color=B,lw=2,label=r"Dropout $\hat\sigma_{\rm epi}$"); ax.plot(dl,a_mcd,"o--",color=B,lw=1.4,alpha=.7,label=r"Dropout $\hat\sigma_{\rm ale}$")
for xv,f_ in zip(dl,frac): ax.text(xv,6,f"{int(f_*100)}%",ha="center",va="bottom",fontsize=8,color="#555")
ax.text(0.15,17,"beyond training max:",ha="left",va="bottom",fontsize=7.5,color="#555")
ax.set_xlabel(r"Temperature shift $\delta$ (s.d., all seven lag days)"); ax.set_ylabel("Mean component estimate (MW)"); ax.set_ylim(0,268); ax.set_title("(c) Temperature shift",loc="left",fontweight="bold"); ax.grid(alpha=.25)
ax3=ax.twinx(); ax3.plot(dl,r_ens,":",color=G,lw=1.4,label=r"Ensemble $R_{\rm epi}$ (right axis)"); ax3.plot(dl,r_mcd,":",color=B,lw=1.4,label=r"Dropout $R_{\rm epi}$ (right axis)"); ax3.set_ylabel(r"$R_{\rm epi}$"); ax3.set_ylim(0,0.233)
h1,l1=ax.get_legend_handles_labels(); h3,l3=ax3.get_legend_handles_labels(); ax.legend(h1,l1,frameon=False,fontsize=7.5,loc="upper left"); ax3.legend(h3,l3,frameon=False,fontsize=7.5,loc="upper right")
plt.tight_layout()
for ext in ["pdf","png"]: plt.savefig(f"fig4_verification_behavioural.{ext}",dpi=220 if ext=="png" else None)
print("ok")
