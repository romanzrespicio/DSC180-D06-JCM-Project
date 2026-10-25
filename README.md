# DSC180 D06 — JCM Project

Team repo. Decision log is in [NOTES.md](NOTES.md).

## Environment

```bash
mamba env create -f environment.yml   # or: conda env create -f environment.yml
conda activate jcm
python check_env.py
```

On DataHub (GPU), also install the CUDA build of jax and check the GPU is visible:

```bash
pip install -U "jax[cuda12]"
python check_env.py --gpu
```

`jax.devices()` should print something like `[CudaDevice(id=0)]`.

### JCM / JEM on NRP (Weeks 4–6)

The DataHub GPU does not work for us, so we run JCM on the NRP JupyterHub
(https://jupyterhub-west.nrp-nautilus.io, UCSD login, ask for 1 GPU).

`pip install jcm jax-esm` from the brief does not work: jax-esm is not on PyPI, and jax-esm needs
one exact jax-gcm commit. [nrp/setup_jcm.sh](nrp/setup_jcm.sh) installs both at the pinned
commits, with CUDA jax:

1. Upload `nrp/setup_jcm.sh` (and any other `nrp/*.py` you need) with the JupyterLab Upload button.
2. In a terminal: `chmod +x ~/setup_jcm.sh && ~/setup_jcm.sh` (about 8 min). The last line must
   print `[CudaDevice(id=0)]`.
3. Select the kernel **Python (jcm-model)**, or use `/tmp/jcm-model/bin/python` in the terminal.

Limits of the JupyterHub server (checked 2026-10-25):
- The home folder is 5 GB, so the env (6.4 GB) goes in `/tmp`. `/tmp` is wiped when the server
  restarts: run `~/setup_jcm.sh` again then. Your home folder stays.
- 8 GB RAM, **1 CPU core**, GPU = GTX 1080 Ti (11 GB). The server stops 1 h after the browser closes.
- Start long runs with `nohup ... &` and add `-u` after `python` so the log updates.

Laptop (CPU only, for small tests and analysis):
`conda create -n jcm-model python=3.12`, then
`pip install -r nrp/requirements.txt` with `jax[cuda12]` changed to `jax` on a Mac.

### Gaussian-process env (Week 3)

The GP emulator (`notebooks/Week3_GaussianProcess.ipynb`) uses GPflow on TensorFlow, which needs
different numpy/ml_dtypes versions than jax. It has its own env so `jcm` stays untouched:

```bash
mamba env create -f environment-gp.yml
conda activate jcm-gp
python -m ipykernel install --user --name jcm-gp --display-name "Python (jcm-gp)"
```

Then pick the **Python (jcm-gp)** kernel in Jupyter. Only tested on macOS (arm64) so far.

## Week 3 baselines: shared prediction format

Each baseline notebook ends by saving its `ssp245` predictions to `results/<model>_ssp245.nc`
(`pattern_scaling`, `gp`, `rf`, `cnn_lstm`), so `notebooks/Week3_Leaderboard.ipynb` (Task 5) can score
every model the same way:

- variables `tas`, `diurnal_temperature_range`, `pr`, `pr90`
- dims `(time, lat, lon)` on the ClimateBench 96 × 144 grid, `time` in years 2015–2100
- precipitation in mm/day (× 86400 from kg m-2 s-1)

The metric code (spatial / global / total NRMSE) is in `metrics.py`; import it to check your scores
while developing. `results/*.nc` files are committed to git so everyone can run the leaderboard.

## Week 4: JCM fixed-SST runs

Status: Task 2 (throughput) and Task 3 (control run) are done (coltranegowan).

### Task 2: throughput

SPEEDY T31L8, 15-min step, 1 simulated year, timed with [nrp/throughput.py](nrp/throughput.py)
(call 1 = compile + run, call 2 = run only). Raw results: `results/throughput_*.json`.

| Hardware | Compile | Run, 1 sim year | Sim years / hour |
|---|---|---|---|
| NRP GTX 1080 Ti (GPU) | 28.6 s | 64.5 s | 55.8 |
| MacBook Air M2 (CPU, 8 cores) | ~6 s | 6–7 min | 9–10 |
| NRP JupyterHub (CPU, 1 core) | 32.1 s | ~16 min | 3.8 |

- The 1-core CPU number is from a 30-day run, scaled (the cost per step is constant).
- The M2 slows down under sustained load (no fan); its 365-day run gave 8.8, its 10-day run 10.0.
- The 11-year control took 754 s on the GPU (about 52 sim years/hour with compile).

### Task 3: control run

- Command: `python -m jcm.main +configuration=speedy-t31` with monthly means, 2000-01-01 to
  2011-01-01. Full resolved config: [nrp/w4-control_config.yaml](nrp/w4-control_config.yaml).
- Forcing: present-day climatology `hf://bundles/t63/forcing_pd.nc` (SST/sea ice: PCMDI-AMIP
  2005–2014; land: ERA5 2005–2014), replayed every year. Init: isothermal rest.
- **CO2 = 420 ppmv** (`jcm.forcing.DEFAULT_CO2_VMR_PPMV`; the bundle has no `co2` variable).
  The comment `# 360.0` in `jcm/physics/forcing/speedy_forcing.py` is out of date.
- JCM uses an analytic ozone profile on SPEEDY's sigma levels (it logs a warning about this).
- **Spin-up:** the global-mean TOA net flux settles within the first year. Use 2001–2010 for
  the analysis, and drop 2000.

Data for the other tasks: [results/w4-control_monthly.nc](results/w4-control_monthly.nc)
(17 MB, made by [nrp/extract_monthly.py](nrp/extract_monthly.py)). 132 months, T31 grid
(48 lat × 96 lon), 8 sigma levels.

| Variable | Meaning | Units |
|---|---|---|
| `rsdt` | incoming SW at TOA | W m-2 |
| `rsnt` | net (absorbed) SW at TOA | W m-2 |
| `rsut` | reflected SW at TOA (`rsdt - rsnt`) | W m-2 |
| `rlut` | OLR | W m-2 |
| `rtnt` | net downward flux at TOA (`rsnt - rlut`) | W m-2 |
| `tas` | near-surface air temperature | K |
| `ts` | surface temperature | K |
| `pr` | total precipitation (× 86400 for mm/day) | kg m-2 s-1 |
| `evap`, `clt`, `ps_norm` | evaporation, cloud cover, surface pressure / p0 | |
| `ta`, `ua`, `va` | **zonal means** of T, u, v, dims (time, level, lat) | K, m s-1 |

`level` is sigma with the **surface first** (0.9375 … 0.0625). Select levels by value
(`ds.sel(level=0.9375)`), never by index. Weight global means by cos(lat).

### Next tasks: how to start

- **ERF runs (0.5×, 2×, 4×, 8× CO2):** the JCM CLI has no CO2 option; CO2 comes from a `co2`
  variable in the forcing file. On NRP:
  ```bash
  /tmp/jcm-model/bin/python ~/make_co2_forcing.py ~/forcing 0.5 2 4 8   # 210, 840, 1680, 3360 ppmv
  mkdir -p ~/runs/w4-co2x4 && cd ~/runs/w4-co2x4
  nohup /tmp/jcm-model/bin/python -u -m jcm.main +configuration=speedy-t31 \
    forcing.file=$HOME/forcing/forcing_pd_co2x4.nc \
    run.total_time=null run.start_time=2000-01-01 run.end_time=2011-01-01 \
    run.save_interval=1 run.output_averages=true run.chunk_days=30 \
    run.monthly_means=true run.save_chunks=false run.checkpoint_path=ckpt.msgpack \
    hydra.run.dir=$HOME/runs/w4-co2x4 > run.log 2>&1 &
  /tmp/jcm-model/bin/python ~/extract_monthly.py ~/runs/w4-co2x4     # when it finishes
  ```
  Set `run.end_time` from the variability task. Each run is ~1.1 GB on disk: run
  `extract_monthly.py`, download the small file, then delete the run folder, because the home
  folder is 5 GB.
- **ERA5:** subset on Casper (NCAR RDA) to t2m, total precipitation, T, u, and the TOA fluxes.
