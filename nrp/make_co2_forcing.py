"""Write copies of the speedy-t31 forcing bundle with a fixed CO2 value.

The JCM CLI has no co2 option: CO2 comes from a scalar `co2` variable (ppmv)
in the forcing file, and the default bundle (hf://bundles/t63/forcing_pd.nc)
has none, so the control runs at jcm.forcing.DEFAULT_CO2_VMR_PPMV (420 ppmv at
the pinned commit; the "# 360.0" comment in speedy_forcing.py is stale). Each copy keeps
the same SST/sea-ice climatology, which is what a fixed-SST ERF run needs.

Usage:
    python make_co2_forcing.py OUT_DIR [MULTIPLE ...]   # default: 0.5 2 4 8
Then run with
    forcing.file=OUT_DIR/forcing_pd_co2x4.nc
"""
import sys
from pathlib import Path

import xarray as xr
from huggingface_hub import hf_hub_download
from jcm.forcing import DEFAULT_CO2_VMR_PPMV as CO2_CONTROL_PPMV

out_dir = Path(sys.argv[1])
multiples = [float(m) for m in sys.argv[2:]] or [0.5, 2.0, 4.0, 8.0]
out_dir.mkdir(parents=True, exist_ok=True)

src = hf_hub_download("climate-analytics-lab/jax-gcm-data", "bundles/t63/forcing_pd.nc",
                      repo_type="dataset")
ds = xr.open_dataset(src).load()
assert "co2" not in ds, "bundle already has co2; check the control value"

for m in multiples:
    out = ds.copy()
    out["co2"] = xr.DataArray(CO2_CONTROL_PPMV * m, attrs={"units": "ppmv"})
    out.attrs["note"] = f"{ds.attrs.get('note', '')}; co2 = {m:g} x {CO2_CONTROL_PPMV:g} ppmv"
    path = out_dir / f"forcing_pd_co2x{m:g}.nc"
    out.to_netcdf(path)
    print(path, float(out["co2"]))
