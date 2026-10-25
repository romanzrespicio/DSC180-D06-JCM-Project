#!/bin/bash
# Install the JCM env on the NRP JupyterHub. Run once per server session:
#   ~/setup_jcm.sh
# The env (6.4 GB, mostly CUDA libraries) goes in /tmp, because the home folder
# is only 5 GB. /tmp is wiped when the server restarts, so run this again then.
# No uv cache: a second copy of the CUDA wheels in /tmp made the server restart.
# Pins: jcm at the jax-gcm commit that jax-esm is tested against (see
# nrp/requirements.txt, which has the same pins for a laptop install).
set -e
[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
if ! /tmp/jcm-model/bin/python -c "import jcm" 2>/dev/null; then
  export UV_NO_CACHE=1
  uv venv -p 3.12 /tmp/jcm-model
  uv pip install --python /tmp/jcm-model/bin/python \
    "jcm @ git+https://github.com/climate-analytics-lab/jax-gcm@f378069090b127badbb158b06e2767f3be8f436c" \
    "jax-esm @ git+https://github.com/climate-analytics-lab/jax-esm@2ae7a1be6c4523785b4c7fb04f7cf20dfff84d35" \
    "jax[cuda12]>=0.10,<0.11" xarray netcdf4 matplotlib cartopy ipykernel
fi
/tmp/jcm-model/bin/python -m ipykernel install --user --name jcm-model --display-name "Python (jcm-model)"
/tmp/jcm-model/bin/python -c "import jax, jcm, jem; print(jax.devices())"
