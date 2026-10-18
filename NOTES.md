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
  the tas global error (the forest's global mean jumps ~1 K between 2080 and 2090,
  because trees can't interpolate between scenarios).
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
