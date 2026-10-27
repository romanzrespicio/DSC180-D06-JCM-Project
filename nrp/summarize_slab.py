"""Shrink a run_slab.py run to two small files for results/ (the monthly files are ~2 MB per year).

    OUT_annual.nc   annual means of every year, every field
    OUT_clim.nc     monthly climatology of the last N years (default 20, or all
                    whole years if fewer)

Usage:
    python summarize_slab.py RUN_DIR OUT [--clim-years 20]
"""
import argparse
from pathlib import Path

import xarray as xr

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("out")
p.add_argument("--clim-years", type=int, default=20)
a = p.parse_args()

files = sorted(Path(a.run_dir, "monthly").glob("*.nc"))
ds = xr.open_mfdataset(files, combine="by_coords").load()
n = ds.time.dt.year.to_series().value_counts()
whole = sorted(int(y) for y in n.index if n[y] == 12)
ds = ds.sel(time=ds.time.dt.year.isin(whole))
mask = ds.ocean_mask.isel(time=0, drop=True) if "time" in ds.ocean_mask.dims else ds.ocean_mask
fields = ds.drop_vars("ocean_mask")

# Annual means weighted by month length.
days = fields.time.dt.days_in_month
annual = (fields * days).groupby("time.year").sum() / days.groupby("time.year").sum()
for v in fields.data_vars:      # the sum skips NaN, so land would read 0: mask it again
    annual[v] = annual[v].where(fields[v].notnull().any("time")).astype("float32")
    annual[v].attrs = fields[v].attrs
clim_years = whole[-a.clim_years:]
clim = fields.sel(time=fields.time.dt.year.isin(clim_years)).groupby("time.month").mean().astype("float32")

enc = lambda d: {v: {"zlib": True, "complevel": 4} for v in d.data_vars}  # noqa: E731
for name, d, extra in [("annual", annual, {}), ("clim", clim, {"clim_years": f"{clim_years[0]}-{clim_years[-1]}"})]:
    d["ocean_mask"] = mask
    d = d.transpose(..., "lat", "lon")
    d.attrs.update(ds.attrs, **extra)
    path = Path(f"{a.out}_{name}.nc")
    d.to_netcdf(path, encoding=enc(d))
    print(path, dict(d.sizes), f"{path.stat().st_size / 1e6:.1f} MB")
