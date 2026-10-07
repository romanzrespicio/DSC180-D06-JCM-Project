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
