"""Put Week 4's present-day forcing on the T31 grid, for the slab components.

The jcm atmosphere reads `hf://bundles/t63/forcing_pd.nc` (192x96) and regrids
it itself (`forcing.interpolate`), but jem's slab ocean, land and sea ice only
read a local file that is already on the model grid. This writes that file,
with jcm's own regridding (`upsample_forcings_ds`), so the slab target SST is
the SST the Week 4 control was forced with, cell for cell.

The same file is used as the atmosphere's forcing file, so the atmosphere and
the slab read one source.

Usage:
    python make_slab_bc.py OUT_DIR
writes OUT_DIR/forcing_pd_t31.nc
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from dinosaur.sigma_coordinates import SigmaCoordinates
from jcm.data.bc.interpolate import upsample_forcings_ds
from jcm.forcing_assembly import _resolve_data_path
from jcm.utils import get_coords

out_dir = Path(sys.argv[1]).expanduser()
out_dir.mkdir(parents=True, exist_ok=True)

src = xr.open_dataset(_resolve_data_path("hf://bundles/t63/forcing_pd.nc")).load()
coords = get_coords(vertical_coords=SigmaCoordinates.equidistant(8), spectral_truncation=31)
ds = upsample_forcings_ds(src, grid=coords.horizontal)
assert ds.sizes["lon"] == 96 and ds.sizes["lat"] == 48, dict(ds.sizes)
for v in ds.data_vars:
    assert np.isfinite(ds[v]).all(), f"{v} has non-finite values after regridding"

ds = ds.transpose("lon", "lat", "time", missing_dims="ignore").astype("float32")
ds.attrs.update(src.attrs)
ds.attrs["history"] = "regridded from hf://bundles/t63/forcing_pd.nc to T31 by nrp/make_slab_bc.py " \
                      "(jcm.data.bc.interpolate.upsample_forcings_ds)"
path = out_dir / "forcing_pd_t31.nc"
ds.to_netcdf(path)
print(path, dict(ds.sizes))
