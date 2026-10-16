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
