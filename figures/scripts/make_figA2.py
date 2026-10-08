import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.titlesize':10.5,'axes.titleweight':'bold','axes.labelsize':9.5,'legend.fontsize':8,'figure.dpi':300,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.3,'grid.linewidth':0.6})
C_G='#1f5f8b'   # GEFCom2014 (dark blue)   -- dataset colours, distinct from the method colours (green=ensemble, blue=dropout)
C_C='#c8742a'   # post-COVID (orange-brown)
xl=['30','60','90','180','359\n(full yr)']; x=np.arange(5)
gef=dict(R=[0.17,0.20,0.25,0.18,0.10], dp=[0.056,0.083,0.069,0.069,0.014], dplo=[0.049,0.071,0.058,0.050,0.011], dphi=[0.064,0.098,0.080,0.092,0.018],
         w=[8.7,11.5,15.5,12.0,5.5], d1=[-6.9,3.3,4.6,0.5,4.9], d1lo=[-15.7,-1.1,1.3,-4.4,2.1], d1hi=[0.6,7.7,6.3,4.7,7.5],
         d2=[0.4,-1.5,-0.3,-0.8,1.5], d2lo=[0.03,-1.8,-0.5,-1.4,1.1], d2hi=[0.9,-1.3,-0.2,0.0,2.0], c=[1.5,4,0.5,1.5,0])
cov=dict(R=[0.42,0.21,0.17,0.09,0.08], dp=[0.130,0.060,0.005,0.013,0.015], dplo=[0.111,0.048,0.002,0.009,0.010], dphi=[0.147,0.073,0.008,0.016,0.020],
         w=[25.3,12.2,9.4,5.2,4.4], d1=[0.8,0.2,-0.7,-0.3,0.3], d1lo=[0.3,-0.2,-1.4,-0.9,-0.2], d1hi=[1.2,0.6,0.2,0.2,0.8],
         d2=[-15.8,-11.8,4.8,-1.8,-0.4], d2lo=[-16.9,-14.2,3.8,-4.4,-1.6], d2hi=[-14.7,-8.7,5.7,1.1,0.3], c=[2,4,0,4,0])
fig,ax=plt.subplots(2,3,figsize=(10,6.6))
a=ax[0,0]
for d,c,l,mk in [(gef,C_G,'GEFCom2014','o'),(cov,C_C,'post-COVID','s')]: a.plot(x,d['R'],marker=mk,ls='-',color=c,label=l,ms=5.5,lw=1.8)
a.axhline(0.06,color='k',ls=':',lw=0.9); a.text(4.1,0.07,'main protocol ($K=10$)',fontsize=7.5,ha='right')
a.set_ylabel('Epistemic share $R_{\\mathrm{epi}}$'); a.set_title('(a) Epistemic share'); a.legend(frameon=False)
a=ax[0,1]
for d,c,mk in [(gef,C_G,'o'),(cov,C_C,'s')]:
    v=np.array(d['dp'])*100; a.errorbar(x,v,yerr=[v-np.array(d['dplo'])*100,np.array(d['dphi'])*100-v],fmt=mk+'-',color=c,ms=5.5,capsize=2.5,lw=1.6)
a.axhline(0,color='k',lw=0.8); a.set_ylabel('$\\Delta$PICP$_{90}$ (percentage points)'); a.set_title('(b) Coverage lost without $\\hat\\sigma_{\\mathrm{epi}}$')
a=ax[0,2]
for d,c,mk in [(gef,C_G,'o'),(cov,C_C,'s')]: a.plot(x,d['w'],marker=mk,ls='-',color=c,ms=5.5,lw=1.8)
a.set_ylabel('Width reduction (%)'); a.set_title('(c) Width lost without $\\hat\\sigma_{\\mathrm{epi}}$')
for a in ax[0]: a.set_xticks(x); a.set_xticklabels(xl,fontsize=8); a.set_xlabel('Training history $M$ (days)')
def forest(a,d,key,color,mk,title,xlab):
    v=np.array(d[key]); lo=np.array(d[key+'lo']); hi=np.array(d[key+'hi']); y=np.arange(5)[::-1]; ex=(lo>0)|(hi<0)
    a.errorbar(v,y,xerr=[v-lo,hi-v],fmt='none',ecolor=color,capsize=2.5,lw=1.4)
    a.plot(v[ex],y[ex],mk,color=color,ms=6); a.plot(v[~ex],y[~ex],mk,color='white',mec=color,mew=1.5,ms=6)
    a.axvline(0,color='k',lw=0.8,ls='--'); a.set_yticks(y); a.set_yticklabels([m.replace('\n',' ') for m in xl],fontsize=8); a.set_title(title); a.set_xlabel(xlab)
forest(ax[1,0],gef,'d1',C_G,'o','(d) D1 triage, GEFCom2014','$\\Delta$MAE of unflagged hours (MW)')
forest(ax[1,1],cov,'d1',C_C,'s','(e) D1 triage, post-COVID','$\\Delta$MAE of unflagged hours (MW)')
a=ax[1,2]; y=np.arange(5)[::-1]
for d,c,mk,off in [(gef,C_G,'o',0.18),(cov,C_C,'s',-0.18)]:
    v=np.array(d['d2']); lo=np.array(d['d2lo']); hi=np.array(d['d2hi']); ex=(lo>0)|(hi<0)
    a.errorbar(v,y+off,xerr=[v-lo,hi-v],fmt='none',ecolor=c,capsize=2.5,lw=1.4)
    a.plot(v[ex],(y+off)[ex],mk,color=c,ms=6); a.plot(v[~ex],(y+off)[~ex],mk,color='white',mec=c,mew=1.5,ms=6)
    for yi,hi_,cc in zip(y+off,hi,d['c']): a.text(hi_+0.7,yi,f'$c^\\star\\!=\\!{cc}$',fontsize=7,va='center',ha='left',color=c)
a.axvline(0,color='k',lw=0.8,ls='--'); a.set_yticks(y); a.set_yticklabels([m.replace('\n',' ') for m in xl],fontsize=8)
a.set_title('(f) D2 reweighting, 10:1'); a.set_xlabel('Cost change vs total policy (%)'); a.set_xlim(-20,12)
ax[1,0].set_ylabel('$M$ (days)')
ax[1,0].text(0.02,0.96,'filled: 95% CI excludes zero',transform=ax[1,0].transAxes,fontsize=7.5,color='0.35',va='top')
plt.tight_layout(w_pad=1.6,h_pad=1.8); plt.savefig('figA_datascarce.png',bbox_inches='tight'); plt.close(); print('ok')
