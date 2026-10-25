"""Combine a JCM run's monthly files into one small netCDF with the Week 4 variables.

A speedy-t31 run writes one chunked_run_monthly_YYYY-MM.nc per month with ~77
variables (1.1 GB for 11 years). This keeps only what the Week 4 analysis needs:
2-D fields in full, and the 3-D fields (ta, ua, va) as zonal means, which is all
the cross-sections and the Hadley cell need. Compressed, so it fits in git.

Usage (on NRP):
    python extract_monthly.py ~/runs/w4-control            # -> ~/runs/w4-control/w4-control_monthly.nc

Vertical: `level` is sigma, SURFACE FIRST (index 0 ~ 0.94 = lowest layer).
Select levels by value (ds.sel(level=0.9375)), never by index.

Meanings (checked in jcm/physics/speedy/physics_data.py at the pinned commit):
    shortwave_rad.fsol   incoming solar at TOA                [W m-2]
    shortwave_rad.ftop   NET downward SW at TOA (absorbed)    [W m-2]
    longwave_rad.ftop    outgoing LW at TOA (OLR)             [W m-2]
    -> reflected SW = fsol - sw ftop;  net TOA (down) = sw ftop - lw ftop
"""
import sys
from pathlib import Path

import xarray as xr

KEEP = {
    "shortwave_rad.fsol": "rsdt",          # incoming SW at TOA
    "shortwave_rad.ftop": "rsnt",          # net SW at TOA
    "longwave_rad.ftop": "rlut",           # OLR
    "surface_flux.t0": "tas",              # near-surface air temperature
    "surface_flux.tsfc": "ts",             # surface temperature
    "surface_exchange.precipitation": "pr",
    "surface_flux.evap": "evap",
    "shortwave_rad.cloudc": "clt",
    "normalized_surface_pressure": "ps_norm",
    "temperature": "ta",
    "u_wind": "ua",
    "v_wind": "va",
}

run_dir = Path(sys.argv[1]).expanduser()
files = sorted(run_dir.glob("chunked_run_monthly_*.nc"))
assert files, f"no monthly files in {run_dir}"

ds = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal", coords="minimal",
                       compat="override")
out = ds[list(KEEP)].rename(KEEP)
for v in out.data_vars:
    out[v] = out[v].astype("float32")
    out[v].attrs["jcm_name"] = next(k for k, n in KEEP.items() if n == v)
out["rsut"] = (out.rsdt - out.rsnt).assign_attrs(units="W m-2", long_name="reflected SW at TOA")
out["rtnt"] = (out.rsnt - out.rlut).assign_attrs(units="W m-2", long_name="net downward flux at TOA")
for v in ["ta", "ua", "va"]:                       # zonal means only
    out[v] = out[v].mean("lon", keep_attrs=True).assign_attrs(zonal_mean="true")
out = out.transpose("time", "level", "lat", "lon", ...)
out.attrs.update(source_run=str(run_dir), n_months=len(files),
                 note="ta/ua/va are zonal means; level is sigma, surface first")

path = run_dir / f"{run_dir.name}_monthly.nc"
out.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in out.data_vars})
print(path, dict(out.sizes), f"{path.stat().st_size / 1e6:.0f} MB")
