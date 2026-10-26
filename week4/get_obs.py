"""Build ERA5 and NorESM2-LM monthly climatologies on the JCM T31 grid (Week 4, Task 4).

No Casper: runs on a laptop, DataHub or NRP.
  ERA5 TOA:  Copernicus CDS "ERA5 monthly averaged data on single levels", 2005-2014,
             1.5 deg (rsdt, rsnt, rlut). Needs a free CDS account, ~/.cdsapirc and the
             dataset licence accepted. Downloaded once to data/ (not in git), 12 MB.
  ERA5 rest: WeatherBench2 6-hourly climatology, 1990-2019, 2.8 deg (public, no account)
             (gs://weatherbench2/datasets/era5-hourly-climatology/): tas, pr, ta, ua, va.
             WeatherBench2 lists mean_top_net_*_radiation_flux too, but every value is NaN
             (checked in the 1.5 deg climatology and the raw 6-hourly copies, 1979-2020),
             hence CDS for the TOA terms.
  NorESM2-LM: CMIP6 historical, Amon, members r1-r3, 2005-2014 (the years of the
             control's SST/land forcing), from the Pangeo CMIP6 archive (gs://cmip6).

Writes results/w4-era5_clim.nc and results/w4-noresm2_clim.nc with the same variable
names and units as results/w4-control_monthly.nc:
  tas, pr, rsdt, rsut, rlut, rsnt, rtnt    (month, lat, lon) on the T31 grid
  ta, ua, va                               (month, plev, lat) zonal means, plev in hPa
  <v>_gm                                   (month) global means on the NATIVE grid
The global means are taken before regridding, so the TOA budget does not depend on
the interpolation. Regridding is bilinear (source grids are as fine or finer than T31).

Usage:  python week4/get_obs.py          (~2 GB read, ~5-10 min on home internet)
Needs:  xarray, dask, zarr, gcsfs, netCDF4, cdsapi
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
ANON = {"token": "anon"}
WB2 = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_{}_equiangular_with_poles_conservative.zarr"
CMIP6 = "gs://cmip6/CMIP6/CMIP/NCC/NorESM2-LM/historical/{member}/Amon/{var}/gn/{version}/"
NORESM_VERSIONS = {"r1i1p1f1": "v20190815", "r2i1p1f1": "v20190920", "r3i1p1f1": "v20190920"}
SECONDS_6H = 6 * 3600
SECONDS_DAY = 86400
CDS_TOA = REPO / "data/era5_toa_monthly_2005-2014.nc"

control = xr.open_dataset(REPO / "results/w4-control_monthly.nc")
T31_LAT, T31_LON = control.lat, control.lon


def global_mean(da):
    w = np.cos(np.deg2rad(da.lat))
    return da.weighted(w).mean(("lat", "lon"))


def to_t31(da):
    """Bilinear interpolation onto the T31 Gaussian grid, periodic in longitude."""
    da = da.sortby("lat")
    if "lon" in da.dims:
        da = da.sortby("lon")
        da = xr.concat([da.isel(lon=[-1]).assign_coords(lon=da.lon[[-1]] - 360), da,
                        da.isel(lon=[0]).assign_coords(lon=da.lon[[0]] + 360)], "lon")
        return da.interp(lat=T31_LAT, lon=T31_LON)
    return da.interp(lat=T31_LAT)


def finish(fields, attrs):
    """fields: name -> native-grid DataArray with dims (month, lat, lon) or (month, plev, lat, lon)."""
    out = {}
    for name, da in fields.items():
        if "plev" in da.dims:
            out[name] = to_t31(da.mean("lon"))  # zonal mean first; NaN below ground is skipped
        else:
            out[f"{name}_gm"] = global_mean(da)
            out[name] = to_t31(da)
    ds = xr.Dataset(out).transpose("month", "plev", "lat", "lon", ...).astype("float32")
    for v in ds.data_vars:
        base = v.removesuffix("_gm")
        ds[v].attrs = dict(control[base].attrs) if base in control else {}
        ds[v].attrs.pop("jcm_name", None)
        ds[v].attrs.pop("zonal_mean", None)
        if v.endswith("_gm"):
            ds[v].attrs["note"] = "cos(lat)-weighted global mean on the native grid"
    ds["plev"].attrs = {"units": "hPa", "long_name": "pressure"}
    ds.attrs.update(attrs, note="ta/ua/va are zonal means; <v>_gm are native-grid global means")
    return ds


def era5_toa():
    """ERA5 monthly-mean TOA fluxes 2005-2014 from CDS, as monthly climatologies in W m-2."""
    if not CDS_TOA.exists():
        import cdsapi
        CDS_TOA.parent.mkdir(exist_ok=True)
        cdsapi.Client().retrieve("reanalysis-era5-single-levels-monthly-means", {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": ["toa_incident_solar_radiation", "top_net_solar_radiation",
                         "top_net_thermal_radiation"],
            "year": [str(y) for y in range(2005, 2015)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": ["00:00"],
            "grid": [1.5, 1.5],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }, str(CDS_TOA))
    ds = xr.open_dataset(CDS_TOA).rename(latitude="lat", longitude="lon")
    assert ds.sizes["valid_time"] == 120
    # monthly means of accumulated fields are mean DAILY accumulations (J m-2)
    clim = (ds[["tisr", "tsr", "ttr"]] / SECONDS_DAY).groupby("valid_time.month").mean()
    return {"rsdt": clim.tisr, "rsnt": clim.tsr,
            "rlut": -clim.ttr}  # ttr is net LW, negative (upward)


def era5():
    # WB2 dayofyear runs 1..366; map it to months with a leap-year calendar
    month = xr.DataArray(pd.date_range("2000-01-01", periods=366).month, dims="dayofyear")

    def monthly(ds, var):
        da = ds[var].rename(latitude="lat", longitude="lon").assign_coords(dayofyear=month.values)
        if "level" in da.dims:
            da = da.rename(level="plev")
        return da.mean("hour").groupby("dayofyear").mean().rename(dayofyear="month")

    clim = xr.open_zarr(WB2.format("128x64"), storage_options=ANON)
    fields = {
        "tas": monthly(clim, "2m_temperature"),
        "pr": monthly(clim, "total_precipitation_6hr") * 1000 / SECONDS_6H,  # m per 6 h -> kg m-2 s-1
        "ta": monthly(clim, "temperature"),
        "ua": monthly(clim, "u_component_of_wind"),
        "va": monthly(clim, "v_component_of_wind"),
    }
    fields = {k: v.compute() for k, v in fields.items()}
    toa = {k: v.drop_vars([c for c in v.coords if c not in v.dims]) for k, v in era5_toa().items()}
    fields.update(toa)
    fields["rsut"] = toa["rsdt"] - toa["rsnt"]
    fields["rtnt"] = toa["rsnt"] - toa["rlut"]
    return finish(fields, {"source": "ERA5: TOA terms from CDS monthly means 2005-2014; tas, pr, ta, "
                                     "ua, va from the WeatherBench2 hourly climatology 1990-2019",
                           "period": "TOA 2005-2014; other fields 1990-2019"})


def noresm2():
    def monthly(var):
        members = []
        for member, version in NORESM_VERSIONS.items():
            da = xr.open_zarr(CMIP6.format(member=member, var=var, version=version),
                              storage_options=ANON)[var]
            da = da.sel(time=slice("2005", "2014"))
            assert da.sizes["time"] == 120, (member, var, da.sizes["time"])
            members.append(da.groupby("time.month").mean())
        da = xr.concat(members, "member").mean("member")
        if "plev" in da.dims:
            da = da.assign_coords(plev=da.plev / 100)  # Pa -> hPa
        return da.drop_vars([c for c in da.coords if c not in da.dims]).compute()

    fields = {v: monthly(v) for v in ["tas", "pr", "rsdt", "rsut", "rlut", "ta", "ua", "va"]}
    fields["rsnt"] = fields["rsdt"] - fields["rsut"]
    fields["rtnt"] = fields["rsnt"] - fields["rlut"]
    return finish(fields, {"source": "NorESM2-LM CMIP6 historical (Pangeo), mean of r1-r3",
                           "period": "2005-2014"})


for name, build in [("era5", era5), ("noresm2", noresm2)]:
    ds = build()
    path = REPO / f"results/w4-{name}_clim.nc"
    ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    gm = {v: round(float(ds[v].mean("month")), 2) for v in ds.data_vars if v.endswith("_gm")}
    print(path.name, f"{path.stat().st_size / 1e6:.1f} MB", gm)
