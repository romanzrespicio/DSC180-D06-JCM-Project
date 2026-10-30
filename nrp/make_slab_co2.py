"""Copies of the slab boundary file (make_slab_bc.py) with a fixed CO2 value, for the
Week 5 abrupt-CO2 runs.

make_co2_forcing.py does the same for Week 4's hf:// bundle; the slab runs read the
T31 file instead, as the atmosphere's forcing and as the slab climatologies. Only
the atmosphere reads `co2` (a scalar, ppmv); SST, sea ice and land are unchanged.

Usage:
    python make_slab_co2.py BC.nc [MULTIPLE ...]     # default: 2 4
writes BC_co2x2.nc, BC_co2x4.nc next to BC.nc
"""
import sys
from pathlib import Path

import xarray as xr
from jcm.forcing import DEFAULT_CO2_VMR_PPMV as CO2_CONTROL_PPMV

bc = Path(sys.argv[1]).expanduser()
multiples = [float(m) for m in sys.argv[2:]] or [2.0, 4.0]

ds = xr.open_dataset(bc).load()
assert "co2" not in ds, "BC file already has co2; check the control value"

for m in multiples:
    out = ds.copy()
    out["co2"] = xr.DataArray(CO2_CONTROL_PPMV * m, attrs={"units": "ppmv"})
    out.attrs["note"] = f"{ds.attrs.get('note', '')}; co2 = {m:g} x {CO2_CONTROL_PPMV:g} ppmv"
    path = bc.with_name(f"{bc.stem}_co2x{m:g}.nc")
    out.to_netcdf(path)
    print(path, float(out["co2"]))
