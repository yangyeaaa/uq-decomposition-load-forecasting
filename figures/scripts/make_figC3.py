import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
# style matched to verification_behavioural2.png: bold panel titles, light grid, green ensemble / blue dropout / grey error
plt.rcParams.update({'font.size':9,'axes.titlesize':10.5,'axes.titleweight':'bold','axes.labelsize':9.5,'figure.dpi':300,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.3,'grid.linewidth':0.6})
GREEN='#2e8b6e'; BLUE='#3b6fb6'; GREY='#4d4d4d'
# label, contrast, lo, hi, auc, alo, ahi, colour, marker
rows=[('Estimation error $\\Delta e$ ($K=10$)',15.0,11.0,19.1,0.64,0.60,0.68,GREY,'D'),
      ('Ensemble, $K=5$ (rep. 1)',4.4,2.8,6.0,0.60,0.56,0.64,GREEN,'s'),
      ('Ensemble, $K=5$ (rep. 2)',4.8,3.2,6.3,0.63,0.59,0.67,GREEN,'s'),
      ('Ensemble, $K=10$',4.6,3.4,5.7,0.66,0.61,0.69,GREEN,'s'),
      ('MC dropout, $p=0.05$',-0.3,None,None,0.49,None,None,BLUE,'o'),
      ('MC dropout, $p=0.10$ (seed 1)',2.0,1.4,2.6,0.59,None,None,BLUE,'o'),
      ('MC dropout, $p=0.10$ (3-seed mean)',0.6,0.2,0.9,0.54,None,None,BLUE,'o'),
      ('MC dropout, $p=0.30$',1.8,None,None,0.55,None,None,BLUE,'o')]
# y positions with gaps between groups
y=np.array([9.0, 7.6,6.6,5.6, 4.2,3.2,2.2,1.2])
fig,(a,b)=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw=dict(width_ratios=[1.35,1]),sharey=True)
def draw(ax,x,lo,hi,yi,col,mk):
    if lo is not None:
        ax.errorbar(x,yi,xerr=[[x-lo],[hi-x]],fmt=mk,color=col,ms=6.5,capsize=3,lw=1.6,zorder=3)
    else:
        ax.plot(x,yi,mk,color=col,ms=6.5,mfc='white',mew=1.6,zorder=3)
for yi,(lab,c,lo,hi,auc,alo,ahi,col,mk) in zip(y,rows):
    draw(a,c,lo,hi,yi,col,mk); draw(b,auc,alo,ahi,yi,col,mk)
a.axvline(0,color='k',lw=0.9,ls='--',zorder=1); a.set_yticks(y); a.set_yticklabels([r[0] for r in rows],fontsize=8.5)
a.set_xlabel('Contrast: thinned $-$ unthinned (MW)'); a.set_title('(a) Response to reduced training support')
a.set_xlim(-2,20.5); a.set_ylim(0.4,9.8)
b.axvline(0.5,color='k',lw=0.9,ls=':',zorder=1); b.text(0.506,9.45,'chance',fontsize=8,color='k')
b.set_xlabel('AUC: thinned vs. unthinned'); b.set_title('(b) Localization of the thinned region'); b.set_xlim(0.44,0.72)
b.tick_params(axis='y',length=0)
# legend for marker semantics
from matplotlib.lines import Line2D
h=[Line2D([],[],marker='D',color=GREY,ls='',ms=6,label='estimation error (reference)'),
   Line2D([],[],marker='s',color=GREEN,ls='',ms=6,label='deep ensemble'),
   Line2D([],[],marker='o',color=BLUE,ls='',ms=6,label='MC dropout'),
   Line2D([],[],marker='o',color=GREY,ls='',ms=6,mfc='white',mew=1.4,label='no bootstrap interval')]
a.legend(handles=h,loc='lower right',frameon=False,fontsize=8)
plt.tight_layout(w_pad=1.5); plt.savefig('figC_control.png',bbox_inches='tight'); plt.close(); print('ok')
