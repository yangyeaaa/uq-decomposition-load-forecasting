"""Fig. 2 (revised): objective-level non-identifiability.
Style matched to the other figures (make_figE.py / make_figA2.py / make_figC3.py):
DejaVu Sans, bold panel titles, light grid, top/right spines off, figure width 10 in.
Colours: grey = free-output construction, gold shades = deep evidential regression (by lambda),
green = deep ensemble (as in Figs. 5-6)."""
import json, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
                     'font.size':9,'axes.titlesize':10.5,'axes.titleweight':'bold','axes.labelsize':9.5,
                     'legend.fontsize':8,'figure.dpi':300,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':0.3,'grid.linewidth':0.6})
GREEN='#2e8b6e'; GREY='#4d4d4d'; GOLD='#b07d2b'
GOLDS=['#e9d3a6','#dcb76f','#c9963f','#b07d2b','#7a541a']   # light -> dark with increasing lambda
# Table 1(a): (R_epi, MAE, ECE, PICP90, init label), from results/logs/free_output_log.txt
# (scripts/free_output_counterexample.py).
fo=[(0.952,109.1,0.031,0.906,'(0, −1)'),(0.036,108.9,0.011,0.880,'(−1, 0)'),(0.208,110.1,0.013,0.875,'(−2, +0.5)')]
# Seed band: MVE MAE over seeds 1-5 of the backbone, results/logs/t1_seed_stability_GEF_log.txt
# (scripts/t1_seed_stability.py).
seed_mu,seed_sd=112.0,1.97
HERE=os.path.dirname(os.path.abspath(__file__))
A=json.load(open(os.path.join(HERE,'..','..','results','json','der_evidential.json')))   # Table 1(b)

fig,ax=plt.subplots(1,3,figsize=(10,3.4),gridspec_kw=dict(width_ratios=[1,1,1.25]))
a=ax[0]
a.axhspan(seed_mu-seed_sd,seed_mu+seed_sd,color=GREY,alpha=0.12,lw=0); a.axhline(seed_mu,color=GREY,lw=0.8,ls='--')
a.text(0.03,seed_mu+seed_sd+0.2,'backbone: mean ± 1 s.d. (5 seeds)',fontsize=7.5,color=GREY)
for r,m,e,p,l in fo:
    a.plot(r,m,'o',color=GREY,ms=7,mec='white',mew=0.8,zorder=3)
    a.annotate(l,(r,m),xytext=(0,-13),textcoords='offset points',ha='center',fontsize=7.5,color=GREY)
a.set_xlim(0,1.05); a.set_ylim(106,116); a.set_xticks([0,0.5,1.0])
a.set_xlabel('Reported epistemic ratio $R_{\\mathrm{epi}}$'); a.set_ylabel('MAE (MW)')
a.set_title('(a) Free-output: accuracy')

a=ax[1]
for (r,m,e,p,l),dy in zip(fo,[9,-14,9]):
    a.plot(r,e,'s',color=GREY,ms=7,mec='white',mew=0.8,zorder=3)
    a.annotate(f'PICP$_{{90}}$={p:.3f}',(r,e),xytext=(0,dy),textcoords='offset points',ha='center',fontsize=7.5,color=GREY)
a.set_xlim(0,1.05); a.set_ylim(0,0.06); a.set_xticks([0,0.5,1.0])
a.set_xlabel('Reported epistemic ratio $R_{\\mathrm{epi}}$'); a.set_ylabel('ECE')
a.set_title('(b) Free-output: calibration')

a=ax[2]
rep=sorted([r for r in A if r['lam']=='reparam'],key=lambda r:r['R_epi'])
a.plot([r['R_epi'] for r in rep],[r['NLL'] for r in rep],'--',color=GREY,lw=1.4,zorder=1)
a.plot([r['R_epi'] for r in rep],[r['NLL'] for r in rep],'o',mfc='white',mec=GREY,ms=6,mew=1.2,zorder=2,label='exact reparameterization')
for lam,c in zip([0.0,1e-3,1e-2,1e-1,1.0],GOLDS):
    rr=[r for r in A if r['lam']==lam]
    a.plot([r['R_epi'] for r in rr],[r['NLL'] for r in rr],'D',color=c,ms=6,mec='white',mew=0.6,zorder=3,label=f'trained, $\\lambda$={lam:g}')
a.plot(0.061,6.334,'s',color=GREEN,ms=7,mec='white',mew=0.8,zorder=4,label='deep ensemble ($K$=10)')
a.set_xlim(0,1.05); a.set_ylim(6.30,6.46); a.set_xticks([0,0.5,1.0])
a.set_xlabel('Reported epistemic ratio $R_{\\mathrm{epi}}$'); a.set_ylabel('Test NLL')
a.set_title('(c) Deep evidential regression')
a.legend(frameon=False,loc='upper left',ncol=2,fontsize=7,handletextpad=0.2,columnspacing=0.8,borderaxespad=0.3)
plt.tight_layout(w_pad=1.6); plt.savefig('nonidentifiability2.png',bbox_inches='tight'); print('ok')
