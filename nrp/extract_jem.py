"""Combine a JEM coupled run's chunk files into one small netCDF.

`python -m jem.main` writes one file per component per chunk (atm-*.nc, ocn-*.nc,
seaice-*.nc, lnd-*.nc), each with every variable of the component. This keeps the
2-D fields that Week 5 needs, with the Week 4 names (see extract_monthly.py) for the
atmosphere. With `coupled_run=long_run` each record is a 30-DAY-WINDOW mean, not a
calendar month (see jem/config/coupled_run/long_run.yaml).

Usage:
    python extract_jem.py RUN_DIR OUT.nc
"""
import sys
from pathlib import Path

import xarray as xr

ATM = {
    "shortwave_rad.fsol": "rsdt",          # incoming SW at TOA
    "shortwave_rad.ftop": "rsnt",          # net SW at TOA
    "longwave_rad.ftop": "rlut",           # OLR
    "surface_flux.t0": "tas",              # near-surface air temperature
    "surface_flux.tsfc": "ts",             # surface temperature
    "surface_exchange.precipitation": "pr",
    "shortwave_rad.cloudc": "clt",
}
OCN = {
    "sea_surface_temperature": "sst",
    "mixed_layer_depth": "mld",
    "total_heat_flux": "hfds_up",          # upward, Q-flux included
    "ice_frazil_melt_energy": "frzmlt",
}
ICE = {"ice_fraction": "sic", "ice_thickness": "sit"}


def load(run_dir, comp, names):
    files = sorted(run_dir.glob(f"{comp}-*.nc"))
    assert files, f"no {comp}-*.nc in {run_dir}"
    ds = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal",
                           coords="minimal", compat="override")
    out = ds[list(names)].rename(names)
    for k, v in names.items():
        out[v] = out[v].astype("float32").assign_attrs(jem_name=f"{comp}:{k}")
    return out


run_dir = Path(sys.argv[1]).expanduser()
out = xr.merge([load(run_dir, "atm", ATM), load(run_dir, "ocn", OCN),
                load(run_dir, "seaice", ICE)], join="exact")
# The slab ocean writes sst = 288.15 K (MASKED_SURFACE_TEMPERATURE) over land and
# frzmlt = 0 exactly; an ocean cell with frzmlt == 0 at every step does not happen.
ocean = (out.frzmlt != 0).any("time").compute()
for v in OCN.values():
    out[v] = out[v].where(ocean)
out["ocean_mask"] = ocean.astype("int8").assign_attrs(long_name="slab-ocean cell (1) or land (0)")
out["rsut"] = (out.rsdt - out.rsnt).assign_attrs(units="W m-2", long_name="reflected SW at TOA")
out["rtnt"] = (out.rsnt - out.rlut).assign_attrs(units="W m-2", long_name="net downward flux at TOA")
out = out.drop_vars([c for c in out.coords if c not in ("time", "lat", "lon")])
out = out.transpose("time", "lat", "lon", missing_dims="ignore")
out.attrs.update(source_run=str(run_dir),
                 note="30-day-window means; ocean fields are NaN over land; hfds_up is upward positive")

path = Path(sys.argv[2])
out.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in out.data_vars})
print(path, dict(out.sizes), f"{path.stat().st_size / 1e6:.1f} MB")
