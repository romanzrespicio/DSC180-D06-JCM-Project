# Decision log

What we asked the AI assistant, what we changed, and how we checked it.
Newest entries at the top. Keep each one short.

Template:

```
## YYYY-MM-DD — <short title> (<who>)
- Asked: what we asked for (tool used, e.g. Claude Code)
- Changed: files / steps that changed, and why
- Checked: independent evidence it's right (leaderboard number, Zenodo file,
  finite-difference check, known physical value, ...). "It ran" doesn't count.
- Open questions:
```

Things to check every time (assistants get these wrong without warning):
- test-set leakage (no fitting or normalizing on test data)
- dimension order, e.g. which vertical index is the surface
- area weighting (cos(lat)) in global means
- units

---

## 2026-10-26 — Week 5 Task 1: first slab-ocean runs (coltranegowan)
- Asked: Claude Code to run the JEM aquaplanet and Earth-like slab configurations, read `SlabOceanModel`, and
  answer what sets the mixed-layer depth and what a Q-flux is.
- Changed: `nrp/extract_jem.py`, `results/w5-task1_{aquaplanet,earth}.nc`, `notebooks/Week5_SlabOcean.ipynb`,
  `figures/week5_*.png`, README Week 5 section. Both runs: 360 days on the laptop CPU (the quick start is 2 days only).
- Checked:
  - The mixed-layer depth in the output equals `60 + (40 − 60) cos³(lat)` to 2e-5 m and does not change in time.
  - Aquaplanet energy budget: TOA +11 W m-2 on an all-ocean planet with C = 1.97e8 J m-2 K-1 gives 1.8 K/yr;
    the measured SST rise is 1.8 K/yr.
  - Relaxed run: the relaxation heat flux from the SST bias and from the slab budget (C dT/dt + F_up) agree
    (−12.7 vs −14.9 W m-2, pattern correlation 0.92), which confirms the sign of `total_heat_flux` (upward positive).
  - The ocean mask from the output (`frzmlt != 0`) equals `lsm < 0.5` in the terrain file at every cell.
  - earth-slab TOA imbalance (+4.9 W m-2) equals the Week 4 fixed-SST control.
- AI got wrong / nearly wrong (Question 4):
  - The first extraction assumed the slab writes 0 K over land; it writes 288.15 K (`MASKED_SURFACE_TEMPERATURE`),
    which would have biased every ocean mean. Caught by the 290 K global mean in a test on the 2-day run.
- Open questions: jem places monthly climatology records at the start of each month (year fraction k/12), so relaxation
  and Q-flux targets lag by about half a month. Does this matter for the Task 2 Q-flux? earth-slab uses jcm's older T30
  climatology, not Week 4's `forcing_pd.nc`; the Q-flux control should use the Week 4 one.

## 2026-10-25 — Week 4 Tasks 4–5: control vs ERA5/NorESM2, TOA variability (annalyellis)
- Asked: Claude Code to evaluate the control climatology (Task 4) and its TOA variability (Task 5)
  without Casper, on a laptop.
- Changed: `week4/get_obs.py`, `results/w4-era5_clim.nc`, `results/w4-noresm2_clim.nc`,
  `notebooks/Week4_Evaluation.ipynb`, `figures/week4_*.png`, README Task 4–5 sections.
  - ERA5 TOA fluxes from Copernicus CDS monthly means, 2005–2014 (free account, `~/.cdsapirc`,
    licence accepted on the dataset page). ERA5 tas/pr/T/u/v from the WeatherBench2 6-hourly
    climatology, 1990–2019 (public, slightly smoothed in time, centred 5 years before the forcing years).
  - NorESM2-LM from Pangeo CMIP6 (`historical`, r1–r3, 2005–2014) instead of ClimateBench: the
    ClimateBench historical outputs are annual anomalies from piControl (global tas 0.55 K) and only tas/pr.
- Checked:
  - ERA5 global TOA: 340.4 in, 97.7 reflected, 242.0 OLR, net +0.71 W m-2; NorESM2 net +0.56. Both are
    plausible observed/CMIP6 imbalances, so units and signs (CDS daily accumulations / 86400, OLR = −ttr) are right.
  - The 190 hPa warm layer is in JCM's raw sigma-level output (tropics: 250.5 K at σ = 0.1875 against
    241.0 K at σ = 0.3125 and 217 K in ERA5), not from our sigma-to-pressure interpolation.
  - The control's levels are evenly spaced: at the pinned commit, `grid: speedy_t31_l8` builds
    `SigmaCoordinates.equidistant(8)` in `jcm/runners.py`; only `get_speedy_coords()` uses SPEEDY's table.
  - JCM's incoming SW of 342.0 W m-2 is SPEEDY's `solc` in `jcm/physics/speedy/physical_constants.py`.
  - TOA annual values are independent (lag-1 −0.05), so SE = σ/√N holds.
- AI got wrong / nearly wrong (Question 4):
  - WeatherBench2's TOA net fluxes are all NaN; the first build silently gave NaN ERA5 TOA terms. Caught
    by printing the global means; switched to CDS.
  - Planned to use the Weeks 1–2 ClimateBench NorESM2 files as the climatology; they are anomalies.
    Caught by checking the global mean (0.55 K, against 287.6 K in piControl).
  - The first summary said the jets were 50–72 m/s in every season; JJA NH is 28 m/s. Corrected after
    checking the table.
  - Before seeing Tasks 2–3 on main, the assistant wrote duplicate run scripts on `week4_annaly`;
    dropped by resetting the branch to main.
- Open questions: does a control on SPEEDY's own sigma levels remove the 190 hPa warm layer and the jet
  bias (this matters for the Task 6 CO2 runs)? Why is OLR about 5 W m-2 low (JCM warns that its analytic
  ozone profile biases clear-sky OLR low; not checked for SPEEDY)? CERES-EBAF would settle the 5 W m-2
  ERA5/NorESM2 disagreement on the TOA components. 2001 may still be spin-up.

## 2026-10-25 — Week 4 Tasks 2–3: NRP env, throughput, control run (coltranegowan)
- Asked: Claude Code to set up JCM + JAX-ESM on NRP (DataHub GPU does not work), time SPEEDY
  (Task 2), and run the control (Task 3).
- Changed: `nrp/setup_jcm.sh` (NRP JupyterHub env), `nrp/requirements.txt` (same pins, laptop),
  `nrp/throughput.py`, `nrp/extract_monthly.py`, `nrp/make_co2_forcing.py`,
  `nrp/w4-control_config.yaml`, `results/throughput_*.json`, `results/w4-control_monthly.nc`,
  README Week 4 section.
  - `pip install jcm jax-esm` from the brief fails: jax-esm is not on PyPI. jcm is pinned to
    the jax-gcm commit in jax-esm's `contract.py` (f378069), jax-esm to 2ae7a1b.
  - JupyterHub home is 5 GB and the CUDA env is 6.4 GB, so the env goes in /tmp (reinstall
    after a restart). With a uv cache also in /tmp, the server restarted; no cache fixed it.
  - The JCM CLI has no CO2 option; CO2 comes from a scalar `co2` in the forcing file, so
    `make_co2_forcing.py` writes copies of the present-day bundle with `co2` set.
  - Control output reduced from 1.1 GB to 17 MB: 2-D fields in full, T/u/v as zonal means.
- Checked:
  - Throughput: call 2 (compiled) faster than call 1 in short runs; GPU 55.8 sim yr/h agrees
    with the control's 754 s for 11 years. 10-day and 365-day M2 runs agree to ~12% (the
    365-day one is slower: the fanless M2 throttles).
  - Variable meanings read from `jcm/physics/speedy/physics_data.py` (sw `ftop` = net SW,
    lw `ftop` = OLR); `precnv + precls` equals `surface_exchange.precipitation`.
  - Control: global-mean TOA terms steady after year 1; tas 287.95 K, P 2.92 mm/day.
  - `ForcingData.from_file` reads the new `co2` (1680 ppmv for 4x).
- AI got wrong / nearly wrong:
  - First assumed a control CO2 of 360 ppmv from the comment in `speedy_forcing.py` (read on
    jax-gcm main). Loading the forcing at the pinned commit gave 420 ppmv
    (`DEFAULT_CO2_VMR_PPMV`); the script now reads the value from jcm.
  - First 1-year M2 timing gave a negative compile time (throttling); used the 10-day run's
    compile time instead.
  - First proposed a custom Docker image + GitHub workflow + Kubernetes Jobs; dropped, the
    JupyterHub was enough for an 11-year run.
- Open questions: the net TOA imbalance in the control is about +4.9 W m-2 (OLR ~236.7);
  check against ERA5/CERES in the evaluation task.

## 2026-10-18 — Week 3 pattern scaling in JAX, Task 6 (annalyellis)
- Asked: Claude Code to re-implement pattern scaling (both versions A and B) in pure JAX and
  check it against the numpy notebook.
- Changed: `notebooks/Week3_PatternScaling_JAX.ipynb`. Patterns fit two ways: closed form
  (`jnp.linalg.lstsq`, jitted) and gradient descent (`jax.grad` per cell, `jax.vmap` over
  13,824 cells, T standardised so plain GD converges). EBM lag written with `jax.lax.scan`.
  No new results files: the JAX predictions equal the saved numpy ones.
- Checked:
  - GD coefficients equal lstsq to 1e-15; predictions equal the saved numpy files to
    ~5e-8 relative (they were saved as float32); all NRMSE scores agree to 3e-8.
  - float64 enabled (`jax_enable_x64`); in float32 the slopes differ from numpy by ~2e-7.
- AI got wrong / nearly wrong (Question 4):
  - The first draft of the notes claimed float32 would differ "at about 1e-6"; measured
    it instead (2e-7) and corrected the text.
- Open questions: for week 8, learn the EBM's tau jointly with its coefficients by gradient
  instead of a grid search.

## 2026-10-18 — Week 3 pattern-scaling emulator, comparison with official (annalyellis)
- Asked: Claude Code to write pattern scaling from scratch (Task 1), answer the brief's question
  about where test-time global-mean T comes from, then compare with the official
  `pattern_scaling_model.ipynb` once ours was finished.
- Changed: `notebooks/Week3_PatternScaling.ipynb`, `results/pattern_scaling_ssp245.nc`,
  `results/pattern_scaling_ebm_ssp245.nc`, `figures/week3_ps_*.png`. Two versions:
  - A (as in the paper): test-time T = true ssp245 ensemble-mean global tas, trailing
    10-year rolling mean. Gets the answer's global trajectory, so it is a reference, not a
    fair emulator.
  - B (fair): T predicted from forcings by a one-box EBM (lagged cumulative CO2, CH4,
    global-mean SO2 and BC), fit on training scenarios; timescale tau = 1 yr chosen by
    leave-one-scenario-out CV over the 6 training scenarios.
- Checked:
  - Global mean of the fitted tas slope is exactly 1 K/K (as it must be).
  - A replica of the official setup reproduces the published pattern-scaling row to
    3 decimals for all four targets (notebook section 9).
  - Leakage audit: patterns, EBM coefficients, tau and scaling use training scenarios only;
    B uses ssp245 inputs (plus historical inputs for the 2014 state) and no ssp245 outputs.
- Result: A = tas 0.320 (= published), dtr 18.55, pr 3.34, pr90 4.13 (published 19.72,
  3.66, 4.46). B = tas 0.383, dtr 18.41, pr 3.45, pr90 4.23. Predicting T costs little on
  ssp245: even the oracle input lags the trend (2.02 K vs 2.07 K over 2080-2100).
- Why ours beats the published row on dtr/pr/pr90: training on hist-GHG and hist-aer
  (the official uses 4 scenarios). Ensemble mean vs one member: no effect; intercept:
  slightly worse. These variants were run after seeing ssp245 scores, so they stay
  diagnostics; the model was not changed to the best-scoring variant.
- AI got wrong / nearly wrong (Question 4):
  - Proposing to drop the intercept after the ablation showed it scores slightly better
    would have been tuning on ssp245; kept the pre-registered choice instead.
  - (Add anything you catch when reviewing the notebook.)
- Found in the official code: `fit_intercept=False`; single members (historical member 2
  because member 1 "has some missing DTR values"); comment "smooth out the internal
  variability otherwise it's cheating", but the smoothed truth still leaks the global trajectory.
- Open questions: the EBM recovers only ~half the hist-aer cooling when hist-aer is left out
  (one linear aerosol term); `Week3_Leaderboard.ipynb` and the GP notebook hardcode
  `REPO = ~/DSC180-D06-JCM-Project`, which fails for clones elsewhere.

## 2026-10-16 — Week 3 random-forest emulator (Coltrane)
- Asked: Claude Code to write an RF emulator without the official notebooks (same
  features as our GP), choose hyperparameters without ssp245, and then compare with
  the official RF.
- Changed: `notebooks/Week3_RandomForest.ipynb`, `results/rf_ssp245.nc`,
  `results/rf_cv_scores.csv` (cached CV search), `figures/week3_rf_importance.png`.
  Hyperparameters from a random search over the paper's search space, scored by
  leave-one-scenario-out CV over the 6 training scenarios (EOFs refit per fold).
  Final model: 5 seeds x 200 trees. Trees are built 25 at a time and averaged,
  because one tree with 13,824 outputs takes 16-100 MB.
- Checked:
  - Features match the official RF (`RF_model_ESEm.ipynb`): CO2, CH4, 5 SO2 PCs,
    5 BC PCs; one full-grid forest per target.
  - Leakage audit: ssp245 is used only for scoring, plots and grid coordinates.
    The CV never sees it.
  - Seed spread is small (tas total 1.48 ± 0.12), so the gap to the paper is real.
- Result: tas total ~1.5 vs 0.40 published (not reproduced); dtr better (16.7 vs
  22.5); pr and pr90 slightly worse. Post-hoc diagnostics (not used to choose the
  model, see notebook section 9): `max_features='sqrt'` causes most of the tas
  spatial error; training on 6 scenarios instead of the paper's 4 causes most of
  the tas global error (the forest's global mean levels off at ~1.4 K after 2050
  and is ~0.5 K too cold over 2080-2100, because trees can't interpolate between
  scenarios).
  - Correction (2026-10-19): an earlier version of this entry said the forest
    "jumps ~1 K between 2080 and 2090". That was the diagnostic with the paper's
    hyperparameters, not the submitted model; caught when plotting the submitted
    model's global mean in `Week3_Presentation.ipynb`.
- AI got wrong / nearly wrong (Question 4):
  - First search space allowed `min_samples_leaf` 1 and 2: ~100 MB per tree, ~20 GB
    for 200 trees on a 16 GB laptop. Caught by timing and measuring one small
    forest before the full run; switched to the paper's space (leaf 4-12).
  - Planned to score CV folds with `metrics.nrmse`, which asserts 2080-2100; the
    historical folds end in 2014. Caught by reading `metrics.py` first; folds use
    an area-weighted RMSE over all years instead.
  - After seeing the poor tas score, the assistant proposed adding a second
    "paper replica" RF to the leaderboard. We dropped it: it was proposed because
    of the ssp245 score, which is the kind of test-set influence the brief warns
    about. The diagnostics stay as text only.
- Found in the official code: the EOF comment says √cos(lat) weights are applied,
  but `Eof(X['BC'])` has no weights; `RF_model_ESEm.ipynb` trains on 4 scenarios
  (no hist-GHG / hist-aer) while `Original_RF_model.ipynb` uses 6.
- Open questions: why the paper replica (4 scenarios, unweighted EOFs) still gets
  tas global 0.100 vs 0.058 published.

## 2026-10-15 — Week 3 GP emulator, shared metrics, comparison with official GP (Roman)
- Asked: Claude Code to write a GP emulator from scratch (without the official
  notebooks), a shared NRMSE module, and a leaderboard notebook; then to compare
  our GP with the official one once ours was finished.
- Changed: `environment-gp.yml` (separate `jcm-gp` env for GPflow/TensorFlow so
  `jcm` keeps working), `metrics.py`, `notebooks/Week3_GaussianProcess.ipynb`
  (section 9 = comparison), `notebooks/Week3_Leaderboard.ipynb`,
  `results/gp_ssp245*.nc`. Our kernel adds a linear CO2 term (warming ~ linear in
  cumulative CO2, Week 1) to a Matérn-5/2 ARD kernel.
- Checked:
  - Perfect prediction scores exactly 0 in `metrics.py`; formula matches the
    paper's scoring code (`analysis_notebooks/paper_figures.ipynb`).
  - Matérn-only GP (no linear term) gives tas global NRMSE 0.0736 vs published
    0.0738, so data, features and metric reproduce the paper. The linear term
    explains our better tas score (total 0.461 -> 0.299).
  - Leakage audit: ssp245 outputs only used for scoring/plots/grid coords; PCA and
    all standardisation use training years only.
- AI got wrong / nearly wrong (Question 4):
  - pip silently upgraded numpy past the pin and fell back to gpflow 2.5; pip's
    tensorflow then segfaulted with conda's pyarrow. Fixed by taking everything
    from conda-forge. Caught by import/smoke tests.
  - Targets were float32 and GPflow needs float64; caught by a short test fit
    before the full run.
  - The assistant had read the official GP code before we asked it to work
    without it; it disclosed this, and we only used the official code for the
    comparison afterwards.
- Found in the official code: the pr90 GP uses `maxiter=35` (others 1000); the
  DTR and pr90 notebooks can't run as saved (mismatched variable names).
- Open questions: effect of per-cell vs single-number target standardisation and
  of area-weighted EOFs (untested); GP predictive std is too wide (98-99% of
  truth within ±2σ, target 95%).

## 2026-10-06 — Repo and environment setup (Roman)
- Asked: Claude Code to create the team repo, NOTES.md, a Python 3.12 env (>= 3.11 required)
  with xarray, dask, netCDF4, cartopy, scikit-learn, jax, flax, optax, and a
  check script.
- Changed: added `environment.yml`, `check_env.py`, `README.md`, `.gitignore`.
  The geo packages come from conda-forge because cartopy needs GEOS/PROJ.
  jax/flax/optax come from pip. On DataHub, `jax[cuda12]` replaces the CPU jax.
- Checked: `python check_env.py` imports everything locally (CPU). Still to do:
  run `python check_env.py --gpu` on DataHub and confirm `jax.devices()` lists a
  CUDA device.
- Open questions: none yet. Climate models get installed in week 4.
