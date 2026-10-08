from common import *
d=prep_daystrat("COVID"); sc=d["sdl"].scale_[0]; r=load_raw("COVID"); vd,vdl=r["val"]; nv=d["nv"]; sl=d["sl"]
sdd=StandardScaler().fit(r["train"][0].reshape(-1,nv)); Xv=sdd.transform(vd.reshape(-1,nv)).reshape(-1,sl,nv)
def ens10(X):
    mus,vs=[],[]
    for k in range(10):
        m=make_model(d,0.1); m.load_state_dict(torch.load(f"{OUT}/COVID_ens{k}.pth",weights_only=True)); rr=predict(m,X,1,False); mus.append(rr["mu"]); vs.append(rr["ale"]**2)
    mus,vs=np.stack(mus),np.stack(vs); return np.sqrt(vs.mean(0)),np.sqrt(mus.var(0,ddof=1))
ale_v,epi_v=ens10(Xv); ale_t,epi_t=ens10(d["te"])
ale=np.concatenate([ale_v,ale_t])*sc/1000; epi=np.concatenate([epi_v,epi_t])*sc/1000; R=epi**2/(epi**2+ale**2)
N=len(R); nd=N//24; dR=np.array([R[i*24:(i+1)*24].mean() for i in range(nd)]); dE=np.array([epi[i*24:(i+1)*24].mean() for i in range(nd)]); dA=np.array([ale[i*24:(i+1)*24].mean() for i in range(nd)])
# day 0 = 2020-01-01 (validation targets), test targets start 2020-04-01 = day 91
np.savez(f"{OUT}/covid_trace_K10.npz",dR=dR,dE=dE,dA=dA)
segs={"01-01..02-29 (pre)":(0,60),"03-01..03-14 (pre, late)":(60,74),"03-15..03-31 (onset)":(74,91),"04-01..04-30 (lockdown)":(91,121),"05-01..06-30":(121,182),"07-01..08-31":(182,244),"09-01..11-06":(244,nd)}
out=open(f"{OUT}/covid_trace_log.txt","w")
for k,(a,b) in segs.items():
    s=f"{k:28s} days {a:3d}-{b:3d}: R_epi={dR[a:b].mean():.3f}  sig_epi={dE[a:b].mean():.2f} MW  sig_ale={dA[a:b].mean():.2f} MW"; print(s); out.write(s+"\n")
out.close()
