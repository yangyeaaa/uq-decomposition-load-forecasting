import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

# ---- style matched to the other figures (bold panel titles, light grid) ----
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
                     'axes.labelsize': 9.5, 'legend.fontsize': 8, 'figure.dpi': 300,
                     'axes.spines.top': False, 'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linewidth': 0.6})
GREEN = '#2e8b6e'; BLUE = '#3b6fb6'; GREY = '#4d4d4d'; LGREEN = '#8fc9b3'; GOLD = '#b07d2b'

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))

# =============================== (a) MC dropout: dropout rate ===============================
p   = [0.05, 0.10, 0.20, 0.30]
epi = [30.5, 39.0, 45.7, 59.2]          # mean epistemic estimate (MW)
nll = [6.403, 6.413, 6.404, 6.435]      # held-out NLL
mae = [110.6, 108.9, 108.9, 108.2]      # test MAE (MW)
R   = [0.048, 0.067, 0.090, 0.122]      # R_epi

a.plot(p, epi, 'o-', color=BLUE, lw=1.8, ms=6, label='$\\hat\\sigma_{\\mathrm{epi}}$ (left axis)')
a.set_xlabel('Dropout rate $p$')
a.set_ylabel('Mean $\\hat\\sigma_{\\mathrm{epi}}$ (MW)')      # black
a.set_ylim(0, 70); a.set_xlim(0.02, 0.33); a.set_xticks(p)

# R_epi labels: first point offset to the right, last point to the left
R_off = [(0, 9, 'center'), (0, 8, 'center'), (0, 8, 'center'), (-10, 4, 'right')]
for x, y_, r, (dx, dy, ha) in zip(p, epi, R, R_off):
    a.annotate(f'$R_{{\\mathrm{{epi}}}}$={r:.3f}', (x, y_), textcoords='offset points',
               xytext=(dx, dy), ha=ha, fontsize=7.5, color=BLUE)

a2 = a.twinx()
a2.plot(p, nll, 's--', color=GREY, lw=1.4, ms=5, label='held-out NLL (right axis)')
a2.set_ylim(6.30, 6.55); a2.set_ylabel('Held-out NLL')          # black
a2.grid(False); a2.spines['top'].set_visible(False)

# MAE labels: first point below-right, last point below-left, middle points below
M_off = [(10, -12, 'left'), (0, -13, 'center'), (0, -13, 'center'), (-4, 8, 'right')]
for x, y_, m, (dx, dy, ha) in zip(p, nll, mae, M_off):
    a2.annotate(f'MAE {m:.1f}', (x, y_), textcoords='offset points',
                xytext=(dx, dy), ha=ha, fontsize=7.5, color=GREY)

h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, loc='upper left', frameon=False)
a.set_title('(a) MC dropout: rate sets the epistemic scale')

# =============================== (b) deep ensemble: ensemble size ===============================
K   = [3, 5, 10, 20]
eb  = [29.3, 30.4, 33.3, 33.8]                                  # Bessel-corrected sigma_epi (MW)
eu  = [e * np.sqrt((k - 1) / k) for e, k in zip(eb, K)]         # uncorrected (sqrt((K-1)/K) factor)
rho = [0.33, 0.59, 0.67]                                        # reproducibility, K = 3, 5, 10
auc = [0.64, 0.73, 0.78, 0.80]                                  # T4 localization AUC

b.plot(K, eb, 's-',  color=GREEN,  lw=1.8, ms=6, label='$\\hat\\sigma_{\\mathrm{epi}}$, Bessel-corrected (left axis)')
b.plot(K, eu, 's--', color=LGREEN, lw=1.4, ms=5, label='$\\hat\\sigma_{\\mathrm{epi}}$, uncorrected (left axis)')
b.set_xscale('log'); b.set_xticks(K); b.set_xticklabels([str(k) for k in K]); b.minorticks_off()
b.set_xlabel('Ensemble size $K$')
b.set_ylabel('Mean $\\hat\\sigma_{\\mathrm{epi}}$ (MW)')      # black
b.set_ylim(0, 60)

b2 = b.twinx()
b2.plot(K[:3], rho, '^:', color=GREY, lw=1.4, ms=6, label='per-input reproducibility, Spearman $\\rho$ (right axis)')
b2.plot(K, auc, 'D:', color=GOLD, lw=1.4, ms=5, label='T4 localization AUC (right axis)')
b2.set_ylim(0.2, 1.0); b2.set_ylabel('$\\rho$ / AUC')
b2.grid(False); b2.spines['top'].set_visible(False)

h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
b.legend(h1 + h2, l1 + l2, loc='upper left', frameon=False)
b.set_title('(b) Ensemble: scale stable, reproducibility improves')

plt.tight_layout(w_pad=2.0)
plt.savefig('figE_ksens.png', bbox_inches='tight')
plt.close(); print('ok')
