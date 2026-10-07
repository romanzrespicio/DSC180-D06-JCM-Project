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
