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
