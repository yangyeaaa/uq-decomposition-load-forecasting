from common import *
d=prep_daystrat("GEF"); sc=d["sdl"].scale_[0]; tmax=d["tr"][:,:,2].max()
out=open(f"{OUT}/mcd_tshift_log.txt","w")
for p in [0.05,0.10,0.20,0.30]:
    m=make_model(d,p); m.load_state_dict(torch.load(f"{OUT}/GEF_p{int(p*100)}_seed1.pth",weights_only=True)); row=[]
    for delta in [0,0.5,1.0,1.5,2.0]:
        X=d["te"].copy(); X[:,:,2]+=delta; r=predict(m,X,40,True); e=r["epi"]*np.sqrt(40/39)
        row.append(f"{e.mean()*sc:6.1f}/{r['ale'].mean()*sc:6.1f}/{np.mean(e**2/(e**2+r['ale']**2)):.3f}")
    s=f"p={p:.2f}  "+"  ".join(row); print(s,flush=True); out.write(s+"\n")
out.write("columns: delta=0,0.5,1,1.5,2 ; sig_epi/sig_ale/R_epi\n"); out.close(); print("DONE")
