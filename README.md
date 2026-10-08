# Evaluating epistemic–aleatoric uncertainty decompositions in probabilistic load forecasting

Code, processed data, logs and result files for

> Y. Yang, Y. Tian, M. Hu, *Evaluating epistemic–aleatoric uncertainty decompositions in
> probabilistic load forecasting: operational value and limits*, submitted to *Applied Energy*.

Probabilistic load forecasts are often reported with their predictive uncertainty split into an
epistemic and an aleatoric part. The paper asks three questions about such a split:

1. Is it determined by the training objective, or can incidental choices move uncertainty
   between the two parts without changing the forecast?
2. What evidence is needed before the two parts can be read according to their names?
3. When does the epistemic part tell an operator something that calibrated total uncertainty
   does not?

The code covers:

- the non-identifiability result (Proposition 2.1) and its demonstration for a free-output
  construction and for deep evidential regression;
- the verification protocol T1–T4, including the semi-synthetic controlled reference and its
  resampling reference;
- the forecasting and calibration case study on GEFCom2014 and a post-COVID demand dataset,
  with statistical, boosting and neural benchmarks;
- the operational analysis in data-rich, data-scarce and shifted regimes.

## Repository layout

```
src/
  diffmod.py                     iTransformer encoder + heteroscedastic Gaussian head (backbone);
                                 also the free-output head of Section 2.2
  decomposition.py               moment-based decomposition (law of total variance) for
                                 MC dropout and deep ensembles
  quantile_regression_baseline.py
scripts/
  common.py                      data pipeline, training loop, scoring helpers, environment variables
  *.py                           one script per experiment (see the reproduction map)
  fig4.py, fig5.py               render Figs. 4 and 7 (fig5.py keeps an earlier figure number)
  fig3.py                        an earlier draft of the framework schematic; not the published Fig. 3
data/
  GEFCom2014/, postCOVID/        processed arrays used in the paper
figures/scripts/                 render Figs. 2, 5, 6 and 8
results/
  logs/                          log of every run reported in the paper and the supplement
  json/                          machine-readable results of the experiments added in revision
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd scripts
```

Four optional environment variables control where things go:

| Variable | Default | Meaning |
|---|---|---|
| `UQ_ROOT` | repository root | location of `src/` and `data/` |
| `UQ_OUT` | `results/out` (`results/out_alt` if `SPLIT=alt`) | checkpoints, predictions and fresh logs |
| `UQ_THREADS` | `1` | PyTorch CPU threads; 1 reproduces the timings of Table 8 |
| `SPLIT` | `hour` | validation split for early stopping and calibration: `hour` (within-day 12 h / 12 h) or `alt` (alternating complete days, Supplementary Section S1) |

`sec62.py` also reads `BLOCK` (default 7), the block length in days of the moving-block bootstrap.

Everything runs on CPU. On one core, a single network takes about 40 s to train on GEFCom2014.
The K=10 ensembles take about 6.5 min on GEFCom2014 and 5 min on post-COVID (Table 8).
Checkpoints (`*.pth`) and prediction bundles (`*.npz`) are not tracked; the scripts regenerate them.

## Data

Each `*_data.npy` file has shape `(N, 7, 6)`. The 7 rows are the same hour on each of the seven
preceding days. Each `*_label.npy` file holds the load at the target hour, 24 h after the last
row. No target-day temperature is included. The column order differs between the two datasets:

| Dataset | Columns 0–5 |
|---|---|
| GEFCom2014 | hour, load, temperature, month, day of month, day of week |
| post-COVID | load, temperature, month, day of month, hour, day of week |

**`data/GEFCom2014/`**: hourly system load (MW) and the average temperature of three weather
stations (°F) from the GEFCom2014 probabilistic load track.

- Training / validation / test = 2012 / 2013 / 2014, with targets from 8 January in each year.
- 8616 / 8592 / 8592 targets.

**`data/postCOVID/`**: hourly demand and temperature (°C) from the post-COVID day-ahead
forecasting competition, data courtesy of BluWave-ai.

- Demand is stored in **kW**. The paper reports MW (see "Units" below).
- Training 8 Jan–31 Dec 2019; validation 1 Jan–31 Mar 2020; test 1 Apr–6 Nov 2020 15:00.
- 8592 / 2184 / 5272 targets.

## Reproduction

Run everything from `scripts/`. Two scripts write checkpoints that the others reuse, so run them
first:

```bash
python step2_table4.py      # writes {GEF,COVID}_ours_seed1.pth (backbone, seed 1) and the Table 4 baselines
python train_ens.py         # K=10 deep ensembles (seeds 10–19) for both datasets
```

Every other experiment script trains whatever it needs if a checkpoint is missing. The figure
scripts `fig4.py` and `fig5.py` only load saved results. Before running `fig4.py`, run
`train_ens.py`, `step3_dropout.py` and `expcd_reducibility_ood.py`. Before running `fig5.py`,
run `train_ens.py`.

### Reading the logs: two conversions

Two conventions in the logs differ from the paper, so a few numbers need converting before
comparison.

**Units.** Post-COVID quantities are logged in kW, as stored in the data. The paper reports MW:

- divide MAE, RMSE, CRPS, pinball loss, IS90 and every σ by 1000;
- subtract ln 1000 ≈ 6.908 from the NLL. A log density changes by this constant when the unit
  changes, so a logged NLL of 11.68 corresponds to 4.772 in the paper. Comparisons between
  methods are unaffected.

MAPE, ECE, PICP90 and R_epi have no unit. GEFCom2014 is in MW throughout.

**Bessel correction.** The paper estimates the ensemble epistemic variance with the factor
1/(K−1) (Eq. (16)). Scripts written before this choice compute it with `np.var` and the factor
1/K. The paper values are corrected as follows:

- `bessel.py` recomputes the affected results from saved predictions:
  - Table 2;
  - the ensemble rows of Tables 3 and 4;
  - the K-sensitivity of Section 4.4;
  - the variance panel of Table S8.
- These scripts apply the K=10 factor √(10/9) to the saved bundle: `online_lag.py`, `sec62.py`,
  `qr_reserve.py`, `fig5.py` and `train_ens.py`.
- These scripts use the 1/(K−1) factor directly: `sec63.py`, `scarce_recalibration.py`,
  `t4control.py`, `t4control2.py`, `t4boot.py`, `recompute_ref_rho.py`, `covid_trace.py`,
  `ensemble_settings.py`, `student_t_mixture.py` and `fig4.py`.
- For Tables S1 and S2, multiply the logged ensemble σ_epi by √(K/(K−1)), which is 1.054 at
  K=10 and 1.118 at K=5, and recompute R_epi per input.

### Reproduction map

Tables and figures without a prefix are in the main paper; S-numbers are in the Supplementary
Material.

| Paper item | Script(s) | Notes |
|---|---|---|
| Table 1(a), Fig. 2(a,b): free-output construction | `free_output_counterexample.py`, `t1_seed_stability.py` | three log-scale initialisations; the seed band in Fig. 2(a) is the MVE MAE over five backbone seeds |
| Table 1(b), Fig. 2(c), Table S9: deep evidential regression | `der_evidential.py` | NIG head on the same backbone; λ × seeds sweep and exact reparameterisation |
| Section 4.1: T1 (seed stability of MC dropout) | `t1_seed_stability.py` | seeds 1–5, p=0.1, M=40 |
| Table 2, Fig. 4: T2 and T3 | `expcd_reducibility_ood.py`, `mcd_tshift.py`, `bessel.py` | nested prefixes with 3 permutations; held-out band; temperature shift |
| Section S3: T2 for dropout, dropout-rate sweep | `step3_dropout.py` | |
| Table 3, Fig. 5, Tables S3–S4: T4 reference test | `step4_synthetic.py` → `t4control.py` → `t4boot.py` → `recompute_ref_rho.py` → `t4_dropout_seeds.py` | `recompute_ref_rho.py` gives the 14-dimensional input-norm correlations and the full-sample AUC |
| Eq. (20), Table S11: resampling reference | `t4_resampling_reference.py` | see the commands below |
| Fig. 6, Section 4.4: dropout rate and ensemble size | `step3_dropout.py`, `expb_ksens.py`, `bessel.py` | |
| Section 4.4, Table S10: ensemble training settings | `ensemble_settings.py` | K=5; weight decay, training dropout, bagging |
| Table 4, Table 8: benchmarks and timings | `step2_table4.py`, `expa_benchmarks.py`, `step1_timing.py`, `crps_grid.py`, `bessel.py` | single core; the K=10 ensemble row is in `bessel_log.txt` |
| Table 5, Fig. 7: static and online calibration | `train_ens.py` → `online_lag.py`, `fig5.py` | `step2b_online_conformal.py` is the earlier single-network version and is not reported in the paper |
| Table 6, Section S4: Student-t likelihood | `student_t_mixture.py` | exact five-component mixture scoring |
| Table S6, Section S2: residual diagnostics, target-day temperature | `expef_diag_t_temp.py` | oracle covariate |
| Table 7, Table S7: data-rich decisions, reserve policies | `sec62.py`, `qr_reserve.py` | `BLOCK=3` and `BLOCK=14` give the block-length sensitivity; `step6_operational.py` is the earlier K=5 version |
| Fig. 8, Table S8: data-scarce regime | `step7_scarce.py`, `sec63.py`, `bessel.py` | K=5; M ∈ {30, 60, 90, 180} days, plus a full-year control (see below) |
| Section 6.3, Table S12: data-scarce, after recalibration | `scarce_recalibration.py` | see the commands below |
| Section 6.4, Fig. 9: post-COVID trace | `covid_trace.py` | prints the trace statistics |
| Section 6.4: dropout share on both datasets | `t1_seed_stability.py GEF COVID` | five seeds per dataset |
| Table S1: alternating-day split | `SPLIT=alt` with `appsplit.py` | apply the Bessel factor above |
| Figs. 2, 4, 5, 6, 7, 8 (rendering) | `../figures/scripts/make_fig2.py`, `fig4.py`, `../figures/scripts/make_figC3.py`, `../figures/scripts/make_figE.py`, `fig5.py`, `../figures/scripts/make_figA2.py` | `make_fig2.py` reads `results/json/der_evidential.json` for panel (c) and embeds the logged values of Table 1(a) and the seed band for panels (a,b); the other `figures/scripts` files embed their numbers |

In the data-scarce scripts, `M=None` (logged as `M=359`) is the full-year control. It trains on
the complete training partition (359 days for GEFCom2014, 358 for post-COVID) and takes its
14-day early-stopping and calibration set from the first 14 days of the validation year.
For M ≤ 180, that set is the 14 days that follow the training window.

Not included:

- the input-support diagnostics of Table S5;
- the rendering of Fig. 9;
- the two schematics, Figs. 1 and 3.

### Commands for the multi-stage scripts

```bash
# T4 resampling reference: R=16 refits per condition, then the K=10 ensemble, then the comparison
python t4_resampling_reference.py ref full 0 16
python t4_resampling_reference.py ref thin 0 16
python t4_resampling_reference.py ens full
python t4_resampling_reference.py ens thin
python t4_resampling_reference.py analyse

# Data-scarce recalibration
python scarce_recalibration.py train GEF 30 60 90 180 None
python scarce_recalibration.py train COVID 30 60 90 180 None
python scarce_recalibration.py analyse

# Ensemble training settings (any subset of: base wd1e-2 wd1e-1 wd1 p0 p0.3 bag)
python ensemble_settings.py

# Free-output construction (indices into the three initialisations; default: all)
python free_output_counterexample.py

# Seed stability of the backbone and of MC dropout (T1, Fig. 2(a) band, Section 6.4)
python t1_seed_stability.py GEF COVID

# Block-length sensitivity of Section 6.2
BLOCK=3 python sec62.py
BLOCK=14 python sec62.py
```

The `ref` stages are independent, so you can run them in parallel by splitting the range
(e.g. `0 8` and `8 16`).

## Results shipped with the code

`results/logs/` holds the log of each run used in the paper. With the two conversions above,
each table value can be traced to a log line. `results/json/` holds the full per-configuration
results of the experiments added in revision:

| File | Content |
|---|---|
| `der_evidential.json` | Table 1(b) and Table S9: per λ and seed, NLL, CRPS, MAE, ECE, PICP90, R_epi, ν, α |
| `ensemble_settings.json` | Table S10: per training configuration, forecast scores and epistemic scale |
| `t4_resampling.json` | Table S11: reference sampling s.d., bias, MSE_ref and ensemble epistemic s.d., by region |
| `scarce_recalibration.json` | Table S12: static and rolling conformal results, with block-bootstrap intervals |

### Run-to-run variation

Training is deterministic on CPU with the seeds and single thread used here. Monte Carlo
dropout draws its inference masks without a fixed seed, so dropout-based epistemic
estimates (for example R_epi in `t1_seed_stability.py`) can differ in the fourth decimal
between runs.

## Software

The reported numbers were produced with Python 3.12, PyTorch 2.13.0, scikit-learn, pyGAM 0.12
and LightGBM 4.7, on a single core of an Intel Xeon 2.10 GHz CPU. Recomputations under
PyTorch 2.14.0 reproduce the values.

`src/diffmod.py` still contains a diffusion module. It is kept only so that older checkpoints
load, and every script calls the model with `use_diffusion=False`.

## Citation

```bibtex
@article{yang2026evaluating,
  author  = {Yang, Ye and Tian, Yuan and Hu, Mengyan},
  title   = {Evaluating epistemic--aleatoric uncertainty decompositions in probabilistic
             load forecasting: operational value and limits},
  journal = {Applied Energy},
  year    = {2026},
  note    = {Submitted}
}
```

`CITATION.cff` contains the same metadata together with the data references.

## License and data sources

The code is released under the MIT License (see `LICENSE`). The license does not cover the
data, which remain under the terms of their original providers. Please cite them when you use
the data:

- **GEFCom2014.** T. Hong, P. Pinson, S. Fan, H. Zareipour, A. Troccoli, R. J. Hyndman,
  "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond",
  *International Journal of Forecasting* 32(3), 896–913, 2016.
- **Post-COVID competition.**
  - M. Farrokhabadi, "Day-Ahead Electricity Demand Forecasting: Post-COVID Paradigm",
    IEEE DataPort, 2020, doi:10.21227/67vy-bs34 (data courtesy of BluWave-ai).
  - M. Farrokhabadi et al., "Day-Ahead Electricity Demand Forecasting Competition: Post-COVID
    Paradigm", *IEEE Open Access Journal of Power and Energy* 9, 185–191, 2022.
