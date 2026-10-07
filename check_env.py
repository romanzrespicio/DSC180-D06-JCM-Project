"""Check that the project environment is set up correctly.

Usage:
    python check_env.py          # just report versions and devices
    python check_env.py --gpu    # also fail if jax can't see a GPU (use on DataHub)
"""
import importlib
import sys

PACKAGES = ["xarray", "dask", "netCDF4", "cartopy", "sklearn", "jax", "flax", "optax"]

print("python", sys.version.split()[0])
assert sys.version_info >= (3, 11), "need Python >= 3.11"

for name in PACKAGES:
    mod = importlib.import_module(name)
    print(f"{name:12s} {getattr(mod, '__version__', '?')}")

import jax
import jax.numpy as jnp

devices = jax.devices()
print("jax.devices():", devices)

# quick sanity check that the device actually computes something
x = jnp.arange(1000.0)
assert float(jnp.sum(x)) == 499500.0

if "--gpu" in sys.argv:
    assert any(d.platform == "gpu" for d in devices), "no GPU visible to jax"
    print("GPU OK")
