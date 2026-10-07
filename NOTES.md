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

## 2026-10-07 — First look at the target files (Roman)
- Checked: `outputs_*.nc` have dims (member, time, lat, lon). time is integer
  years, lat runs -90..90 (96 points), lon 0..357.5 (144 points, 2.5 deg).
- Variables have **no attrs (no units)**. Mean tas in historical is about 0.25,
  so targets look like anomalies (vs piControl?), not absolute values. pr is
  about 3.5e-9, which fits kg m-2 s-1 (x86400 for mm/day). TODO: confirm the
  baseline and units against the paper and `prepare_data.py`.
- Members: historical, hist-GHG, hist-aer, ssp370, ssp245 have 3;
  **ssp126 and ssp585 have only 1**, so no spread for those in task 2.
- Open question: historical ends in 2014, so the task 1 "2005–2015 mean" either
  uses 2005–2014 only or appends 2015 from ssp245. Decide and record here.

## 2026-10-07 — Downloaded ClimateBench data (Roman)
- Source: Zenodo record 7064308, ClimateBench v1.0.0. Files `train_val.tar.gz`
  and `test.tar.gz`; `CMIP6.zip` not needed for tasks 1–3.
- Location on DataHub: `~/data/climatebench/` (kept outside the repo).
- Checked: downloaded byte counts match Zenodo (839129430 and 74442376).
  Zenodo md5: train_val `9bf0ebd9662667cc1ed78fd3eb5cad4e`,
  test `a09a3ddc8c24e1d6d4b1ee4be5936510`.
- Note: ssp245 is the test scenario. Plotting it is fine, but never fit or
  normalize on it.

## 2026-10-06 — Repo and environment setup (Roman)
- Asked: Claude Code to create the team repo, NOTES.md, a Python 3.12 env (>= 3.11 required)
  with xarray, dask, netCDF4, cartopy, scikit-learn, jax, flax, optax, and a
  check script.
- Changed: added `environment.yml`, `check_env.py`, `README.md`, `.gitignore`.
  The geo packages come from conda-forge because cartopy needs GEOS/PROJ.
  jax/flax/optax come from pip. On DataHub, `jax[cuda12]` replaces the CPU jax.
- Checked: `python check_env.py` imports everything locally (CPU). On DataHub
  (Group D06 scipy-ml-notebook, 1 GPU), env at `~/envs/jcm` plus
  `jax[cuda12]`: `check_env.py --gpu` passed with jax 0.11.2,
  `jax.devices()` = `[CudaDevice(id=0)]`.
- Gotcha: on DataHub, `conda activate` didn't take, so the first
  `pip install "jax[cuda12]"` went into base Python's `~/.local` and pulled in
  numpy 2.x, which conflicts with base tensorflow/torch. We uninstalled those
  packages again (base numpy is back to 1.26.4). Fix: call `~/envs/jcm/bin/python`
  directly instead of activating.
- Open questions: none yet. Climate models get installed in week 4.
