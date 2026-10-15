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
