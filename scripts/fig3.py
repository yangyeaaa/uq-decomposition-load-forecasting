import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams["font.family"]="DejaVu Sans"; plt.rcParams["mathtext.fontset"]="dejavusans"
B,G,O,P="#3b6aa5","#3a8f6e","#d9641e","#5d3f9a"
fig=plt.figure(figsize=(17,7.2)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1500); ax.set_ylim(640,0); ax.axis("off")

def box(x,y,w,h,ec="#666",fc="#fff",lw=1.2,r=6,ls="-"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",ec=ec,fc=fc,lw=lw,ls=ls))
def panel(x,y,w,h,ec="#c9c9d0"): box(x,y,w,h,ec=ec,fc="#f7f7f9",lw=1,r=8)
def T(x,y,s,size=11,color="#222",bold=False,ha="center",it=False):
    ax.text(x,y,s,fontsize=size,color=color,ha=ha,va="center",fontweight="bold" if bold else "normal",fontstyle="italic" if it else "normal")
def arrow(x1,y1,x2,y2,color="#333",lw=1.5):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,color=color,lw=lw))

# headers
for x,s in [(120,"1  Input window"),(370,"2  Shared forecaster"),(740,"3  Two mechanisms for $Q$"),(1082,"4  Common decomposition"),(1355,"5  Verification protocol")]:
    T(x,30,s,13,bold=True)
for x in [235,510,960,1205]: ax.plot([x,x],[45,600],color="#ccc",ls=(0,(4,4)),lw=1)

# 1 input
panel(25,150,190,300)
T(120,178,r"$X\in\mathbb{R}^{7\times 6}$",12,bold=True)
T(120,200,"same hour $h$ on the seven\npreceding days $d\\!-\\!6,\\dots,d$",9.5,"#444")
ax.add_patch(Rectangle((50,232),140,140,fc="#fff",ec="#999",lw=0.8))
for i in range(1,7): ax.plot([50,190],[232+20*i]*2,color="#999",lw=0.6)
for j in range(1,6): ax.plot([50+140/6*j]*2,[232,372],color="#999",lw=0.6)
for j,l in enumerate(["load","temp","hour","mon","day","wd"]): T(50+140/6*(j+.5),382,l,6.3,"#555")
T(44,242,"$d\\!-\\!6$",8,"#555",ha="right"); T(44,362,"$d$",8,"#555",ha="right")
T(120,405,"no target-day temperature",9.5,"#444"); T(120,425,"target: load at hour $h$ of day $d\\!+\\!1$\n(lead time 24 h)",9.5,"#444")
arrow(215,300,262,300)

# 2 forecaster
panel(265,150,215,300)
box(285,170,175,118)
T(372,190,r"iTransformer encoder $g_\psi$",11,bold=True); T(372,208,"attention across variates",9.5,"#444")
for k,l in enumerate(["layer 1","layer 2","representation $h$"]):
    box(300,222+22*k,145,16,ec="#8aa4c8",fc="#e8eef7",lw=1,r=3); T(372,230+22*k,l,9,"#334")
arrow(372,290,372,312)
box(285,315,175,128)
T(372,335,r"Heteroscedastic head $q_\omega$",11,bold=True); T(372,352,"one shared hidden layer",9.5,"#444")
T(330,378,r"$\hat\mu$",12,G); T(415,378,r"$\log\hat\sigma^2_{\mathrm{ale}}$",12,O)
import numpy as np
xx=np.linspace(-1,1,60)
ax.plot(330+18*xx,410-18*np.exp(-4*xx**2),color=G,lw=1.5); ax.plot(415+18*xx,410-22*np.exp(-6*xx**2),color=O,lw=1.5)
T(372,424,"trained by heteroscedastic NLL only\nno epistemic output, no auxiliary loss",9,"#444")
arrow(480,300,535,300); T(507,288,r"$f_\theta$",10,"#444")

# 3 mechanisms
panel(540,80,400,215,ec=B)
T(740,103,"(a) Monte Carlo dropout",11.5,B,bold=True)
T(740,121,"one trained network $\\cdot$ dropout rate $p$ in training, kept at inference",9.2,"#444")
for x,l,s in [(600,"mask 1",r"$\theta^{(1)}$"),(705,"mask 2",r"$\theta^{(2)}$"),(840,"mask $M$",r"$\theta^{(M)}$")]:
    box(x,150,90,60,ec=B); T(x+45,172,l,9.5,"#444"); T(x+45,193,s,12)
T(820,183,"…",12)
T(740,240,"$S=M$ stochastic passes of the same weights",9.5,"#444")
T(740,258,"$Q$ induced by dropout masks; magnitude depends on $p$",9.2,B)
T(740,279,"passes T1–T3, fails T4 (Section 4.4)",9.5,"#444",it=True)

panel(540,310,400,215,ec=G)
T(740,333,"(b) Deep ensemble",11.5,G,bold=True)
T(740,351,"$K$ networks trained independently $\\cdot$ different seeds and data orderings",9.2,"#444")
for x,l,s in [(600,"network 1",r"$\theta^{(1)}$"),(705,"network 2",r"$\theta^{(2)}$"),(840,"network $K$",r"$\theta^{(K)}$")]:
    box(x,380,90,60,ec=G); T(x+45,402,l,9.5,"#444"); T(x+45,423,s,12)
T(820,413,"…",12)
T(740,470,"$S=K$ deterministic passes, one per member",9.5,"#444")
T(740,488,"$Q$ = variation over independent runs; finite $K$ adds sampling noise",9.2,G)
T(740,509,"passes T1–T4 (adopted, $K=10$)",9.5,"#444",it=True)
T(740,560,r"each realization $\theta^{(i)}\sim Q$ yields $(\hat\mu^{(i)},\,\hat\sigma^{2,(i)}_{\mathrm{ale}})=f_{\theta^{(i)}}(X),\quad i=1,\dots,S$",10.5,"#222")
arrow(940,187,985,262,color=B); arrow(940,417,985,343,color=G)

# 4 decomposition
panel(975,200,215,210)
T(1082,224,"Same equations for both",11.5,bold=True)
T(1082,256,r"$\bar\mu=\frac{1}{S}\sum_i\hat\mu^{(i)}$",12.5)
T(1082,290,r"$\hat\sigma^2_{\mathrm{ale}}=\frac{1}{S}\sum_i\hat\sigma^{2,(i)}_{\mathrm{ale}}$",12.5,O)
T(1082,326,r"$\hat\sigma^2_{\mathrm{epi}}=\frac{1}{S-1}\sum_i(\hat\mu^{(i)}-\bar\mu)^2$",12.5,G)
T(1082,360,r"$\hat\sigma^2_{\mathrm{total}}=\hat\sigma^2_{\mathrm{ale}}+\hat\sigma^2_{\mathrm{epi}}$",12.5)
T(1082,390,r"law of total variance; $R_{\mathrm{epi}}$ per input",9,"#444")
arrow(1190,305,1228,305)

# 5 protocol
panel(1232,105,245,410,ec="#8a6bbf")
T(1354,135,"Do the components mean what\ntheir names say?",9.5,P,bold=True)
for y,h,t,s,lw in [(162,58,"T1  Stability across runs","same split under different seeds",1.2),(232,58,"T2  Reduction with data","nested subsets, replicated",1.2),
                   (302,58,"T3  Growth on unfamiliar inputs","held-out temp. band; temp. shift",1.2),(372,74,"T4  Known reference","semi-synthetic: known $\\sigma(x)$,\ndeliberately thinned input region",2)]:
    box(1250,y,210,h,ec="#8a6bbf" if lw<2 else P,lw=lw); T(1355,y+20,t,10.5,bold=True); T(1355,y+42 if h<70 else y+47,s,9,"#444")
T(1355,437,"decisive for T1–T3 passers",9,"#444",it=True)
T(1355,472,"T1–T3: necessary conditions",9.5,P); T(1355,488,"T4: agreement with a known answer",9.5,P)

T(750,612,"Calibration of the total uncertainty (Section 5.2) is applied afterwards to the intervals and does not enter the decomposition; the components are reported at their raw scale.",9.5,"#444")
for ext in ["pdf","svg","png"]: fig.savefig(f"fig3_framework.{ext}",dpi=220 if ext=="png" else None)
print("ok")
