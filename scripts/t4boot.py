from common import *
from sklearn.metrics import roc_auc_score
from step4_synthetic import S, dd, Xte, sparse
d=prep_daystrat("GEF"); sc=d["sdl"].scale_[0]
base=make_model(d,0.1); base.load_state_dict(torch.load(f"{OUT}/GEF_ours_seed1.pth",weights_only=True))
f_true=lambda X: predict(base,X,1,False)["mu"]
def sigma_true(X): zT=X[:,-1,2]; return 0.12+0.25/(1+np.exp(-2*zT))
rng2=np.random.RandomState(123); full={}
for k in ["tr","es","cal","te"]:
    f=f_true(d[k]); s=sigma_true(d[k]); full[k]=(d[k], f+s*rng2.randn(len(f)), f, s)
dfull=dict(d); dfull.update(tr=full["tr"][0],trl=full["tr"][1],es=full["es"][0],esl=full["es"][1],cal=full["cal"][0],call=full["cal"][1],te=full["te"][0],tel=full["te"][1])
f_te=full["te"][2]
def load(path,dset,p=0.1):
    m=make_model(dset,p); m.load_state_dict(torch.load(path,weights_only=True)); return m
mus_t=np.stack([predict(load(f"{OUT}/syn_ens{k}.pth",dd),Xte,1,False)["mu"] for k in range(10)])
mus_f=np.stack([predict(load(f"{OUT}/synfull_ens{k}.pth",dfull),Xte,1,False)["mu"] for k in range(10)])
D={}
D["ens K=10"]=np.sqrt(mus_t.var(0,ddof=1))-np.sqrt(mus_f.var(0,ddof=1))
D["ens K=5 (1-5)"]=np.sqrt(mus_t[:5].var(0,ddof=1))-np.sqrt(mus_f[:5].var(0,ddof=1))
D["ens K=5 (6-10)"]=np.sqrt(mus_t[5:].var(0,ddof=1))-np.sqrt(mus_f[5:].var(0,ddof=1))
D["Delta e K=10"]=np.abs(mus_t.mean(0)-f_te)-np.abs(mus_f.mean(0)-f_te)
mt=load(f"{OUT}/syn_p10.pth",dd); mf=load(f"{OUT}/synfull_p10.pth",dfull)
D["MCD p=0.1"]=predict(mt,Xte,40,True)["epi"]-predict(mf,Xte,40,True)["epi"]
N=len(Xte); day=np.arange(N)//24; ndays=day.max()+1; rng=np.random.RandomState(0); B=1000
out=open(f"{OUT}/t4boot_log.txt","w")
for lab,Dv in D.items():
    aucs,cons=[],[]
    for b in range(B):
        days=rng.randint(0,ndays,ndays); idx=np.concatenate([np.where(day==dd_)[0] for dd_ in days])
        s=sparse[idx]; x=Dv[idx]
        if s.sum()==0 or (~s).sum()==0: continue
        aucs.append(roc_auc_score(s,x)); cons.append((x[s].mean()-x[~s].mean())*sc)
    aucs,cons=np.array(aucs),np.array(cons)
    line=f"{lab:16s} AUC={roc_auc_score(sparse,Dv):.3f} [{np.percentile(aucs,2.5):.3f}, {np.percentile(aucs,97.5):.3f}]   contrast={(Dv[sparse].mean()-Dv[~sparse].mean())*sc:+.1f} MW [{np.percentile(cons,2.5):+.1f}, {np.percentile(cons,97.5):+.1f}]   P(AUC<=0.5)={np.mean(aucs<=0.5):.3f}"
    print(line,flush=True); out.write(line+"\n")
out.write("block bootstrap by target day, B=1000\n"); out.close(); print("DONE")
