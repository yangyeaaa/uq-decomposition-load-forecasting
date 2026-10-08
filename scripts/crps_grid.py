from common import *
import json, warnings; warnings.filterwarnings("ignore")
from expa_benchmarks import design, fit_lgb_quantile, COLS
import quantile_regression_baseline as QR
GRID=np.round(np.arange(0.025,0.976,0.025),3)
def crps_grid(Q_src, tau_src, y, sdl):
    sc,mn=sdl.scale_[0],sdl.mean_[0]; Y=y*sc+mn; Qs=np.sort(Q_src,1)*sc+mn
    Q=np.stack([np.interp(GRID,tau_src,Qs[i]) for i in range(len(Y))])
    pin=np.array([np.maximum(t*(Y-Q[:,j]),(t-1)*(Y-Q[:,j])) for j,t in enumerate(GRID)]).T   # N x 39
    return np.mean(2*np.trapezoid(pin,GRID,axis=1))
out=open(f"{OUT}/crps_grid_log.txt","w")
for name in ["GEF","COVID"]:
    d=prep_daystrat(name); c=COLS[name]; y=d["tel"]
    P=np.load(f"{OUT}/{name}_step2_preds.npz"); Qn=P["qr_Q"]; tau_n=np.array(QR.QGRID)
    cn=crps_grid(Qn,tau_n,y,d["sdl"])
    Xtr=design(d["tr"],c); Xte=design(d["te"],c); Xes=design(d["es"],c)
    Ql=fit_lgb_quantile(Xtr,d["trl"],Xte,GRID,Xes,d["esl"]); cl=crps_grid(Ql,GRID,y,d["sdl"])
    s=f"{name}: neural QR CRPS(grid)={cn:.2f}   LightGBM quantile CRPS(grid)={cl:.2f}   (unit: standardized*scale -> MW for GEF, kW for COVID)"
    print(s); out.write(s+"\n")
out.close()
