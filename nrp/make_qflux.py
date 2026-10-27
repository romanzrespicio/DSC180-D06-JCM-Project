"""Q-flux climatology for the slab ocean, from a relaxed run made by run_slab.py.

In the relaxed run, each daily step of `SlabOceanModel` is the implicit update

    T1 = Tc(t1) + (T0 - Tc(t0) - dt F / C) / (1 + dt / tau),

with C = rho cp h, F the upward surface heat flux and Tc the SST climatology
(floored at freezing, linearly interpolated between records at year fractions k/12).
Rearranged, the heat the relaxation adds to the mixed layer over that step is exactly

    R = C (Tc(t1) - Tc(t0)) / dt - C (T1 - Tc(t1)) / tau        [W m-2, positive warms]

R is linear in the output SST T1, so its calendar-month mean follows exactly from
the monthly-mean SST and the climatology sampled at the same daily steps. This is
the Q-flux: with Q = R and the same surface fluxes, a free slab follows the relaxed
run's SST. Its annual mean is not the mean upward flux F: where the SST would go
below freezing, the slab clamps it at freezing (the deficit becomes sea ice), which
heats the mixed layer, and R balances that heat as well (about -4 W m-2 of the
ocean mean comes from cells with sea ice).

The slab reads `qflux` as values AT year fractions k/12 (month starts) and
interpolates linearly, so the monthly means M_m are written as Q_k = (M_{k-1} + M_k) / 2.
This keeps the annual mean exactly and the phase right, and smooths the seasonal
cycle slightly (the interpolated monthly means are (M_{m-1} + 2 M_m + M_{m+1}) / 4).

Usage:
    python make_qflux.py RELAX_RUN_DIR BC.nc OUT.nc [--skip-years 1]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RHO, CP = 1025.0, 3985.0          # jem.constants ocean_density, ocean_specific_heat_capacity
T_FREEZE = 271.35                 # checked against jem.constants below when jem is importable
TAU = 30 * 86400.0                # earth-slab's relaxation_time
DT = 86400.0                      # coupling step (jem/config/coupling/daily.yaml)


def year_fraction(t):
    """jcm.date.fraction_of_year_elapsed for pandas timestamps."""
    t = pd.DatetimeIndex(t)
    days = np.where(t.is_leap_year, 366, 365)
    return np.asarray((t.dayofyear - 1 + (t - t.normalize()).total_seconds() / 86400.0) / days)


def cyclic_linear(x, data):
    """jem.utils.cycles.evaluate_cyclic_linear, for many x at once; data (..., 12)."""
    pos = np.mod(x, 1.0) * 12
    i0 = np.floor(pos).astype(int) % 12
    a = pos - np.floor(pos)
    return (1 - a) * data[..., i0] + a * data[..., (i0 + 1) % 12]


def climatology_terms(sst_clim, year):
    """Calendar-month means of Tc(t1) and of (Tc(t1) - Tc(t0)) / dt over the daily steps of `year`."""
    t0 = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    t1 = t0 + pd.Timedelta(days=1)
    mid_month = (t0 + pd.Timedelta(hours=12)).month.values
    c0, c1 = cyclic_linear(year_fraction(t0), sst_clim), cyclic_linear(year_fraction(t1), sst_clim)
    tc_end = np.stack([c1[..., mid_month == m].mean(-1) for m in range(1, 13)])
    dtc = np.stack([((c1 - c0)[..., mid_month == m] / DT).mean(-1) for m in range(1, 13)])
    return tc_end, dtc                                      # (12, lon, lat)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("relax_run_dir")
    p.add_argument("bc")
    p.add_argument("out")
    p.add_argument("--skip-years", type=int, default=1, help="spin-up years to drop")
    a = p.parse_args()

    try:
        from jem import constants
        assert (constants.ocean_density, constants.ocean_specific_heat_capacity) == (RHO, CP)
        assert abs(constants.seawater_freezing_point_K - T_FREEZE) < 1e-6, constants.seawater_freezing_point_K
    except ImportError:
        pass

    files = sorted(Path(a.relax_run_dir, "monthly").glob("*.nc"))
    # "minimal": ocean_mask (lat, lon) is in every yearly file and must not be stacked along time
    run = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal").load()
    assert run.attrs["mode"] == "relax", run.attrs["mode"]
    n = run.time.dt.year.to_series().value_counts()
    years = sorted(int(y) for y in n.index if n[y] == 12)[a.skip_years:]   # whole years only
    assert years, "no whole years after the spin-up"
    run = run.sel(time=run.time.dt.year.isin(years))

    bc = xr.open_dataset(a.bc)
    sst_clim = np.maximum(bc.sst.transpose("lon", "lat", "time").values.astype(np.float64), T_FREEZE)
    ocean = run.ocean_mask.values.astype(bool)              # (lat, lon)
    heat_cap = RHO * CP * run.mld.isel(time=0).values       # (lat, lon), NaN over land

    r, target = [], []
    for y in years:
        tc_end, dtc = climatology_terms(sst_clim, y)        # (12, lon, lat)
        tc_end, dtc = tc_end.transpose(0, 2, 1), dtc.transpose(0, 2, 1)
        t1 = run.sst.sel(time=str(y)).values                # (12, lat, lon)
        r.append(heat_cap * dtc - heat_cap * (t1 - tc_end) / TAU)
        target.append(tc_end)
    monthly = np.nanmean(np.stack(r), axis=0)               # (12, lat, lon), calendar months
    target = np.where(ocean, np.mean(target, axis=0), np.nan)
    monthly = np.where(ocean, monthly, 0.0)
    nodes = 0.5 * (np.roll(monthly, 1, axis=0) + monthly)   # value at the start of month k

    w = np.cos(np.deg2rad(run.lat.values))[:, None] * ocean
    gm = lambda f: float((f * w).sum() / w.sum())  # noqa: E731
    f_up = run.hfds_up.mean("time").values
    print(f"years {years[0]}-{years[-1]}; ocean-mean Q {gm(monthly.mean(0)):+.2f} W m-2 "
          f"(annual-mean upward surface flux {gm(np.nan_to_num(f_up)):+.2f}; in a steady run they "
          f"differ by the heat the freezing clamp adds where ice forms)")
    print(f"node values keep the annual mean: {gm(nodes.mean(0)):+.4f} vs {gm(monthly.mean(0)):+.4f}")

    out = xr.Dataset(
        {"qflux": (("time", "lat", "lon"), nodes.astype(np.float32),
                   {"units": "W m-2", "long_name": "slab-ocean Q-flux, positive warms the mixed layer",
                    "note": "values at month starts (year fraction k/12), zero over land"}),
         "qflux_monthly_mean": (("time", "lat", "lon"), monthly.astype(np.float32),
                                {"units": "W m-2", "long_name": "calendar-month mean relaxation heat source"}),
         "sst_target": (("time", "lat", "lon"), target.astype(np.float32),
                        {"units": "K", "long_name": "calendar-month mean of the SST climatology as the slab "
                                                    "sees it (Week 4 forcing_pd sst, floored at freezing, "
                                                    "interpolated between month-start records)"}),
         "lsm": (("lat", "lon"), bc.lsm.transpose("lat", "lon").values.astype(np.float32),
                 {"long_name": "land fraction of the forcing file (T31)"}),
         "icec_max": (("lat", "lon"), bc.icec.max("time").transpose("lat", "lon").values.astype(np.float32),
                      {"long_name": "maximum monthly sea-ice concentration of the forcing file"})},
        coords={"time": pd.date_range("2000-01-01", periods=12, freq="MS"),
                "lat": run.lat.values, "lon": run.lon.values},
        attrs={"source_run": str(Path(a.relax_run_dir).resolve()), "years": f"{years[0]}-{years[-1]}",
               "bc": str(Path(a.bc).resolve()), "made_by": "nrp/make_qflux.py"})
    out.to_netcdf(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
