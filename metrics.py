"""ClimateBench evaluation metrics (Week 3, Task 5), shared by every baseline notebook.

Scores are computed over 2080-2100 against the ssp245 ensemble mean, with cos(latitude) area weights:

    spatial = area-weighted RMSE between the 2080-2100 mean maps          / |global mean of truth|
    global  = RMSE over the years between the yearly global-mean series   / |global mean of truth|
    total   = spatial + 5 * global

Predictions are xarray objects with dims (time, lat, lon), time in years, precipitation in mm/day.
Only numpy and xarray are needed, so this works in both the jcm and jcm-gp envs.
"""
import numpy as np
import xarray as xr

TARGETS = ["tas", "diurnal_temperature_range", "pr", "pr90"]
PERIOD = slice(2080, 2100)


def load_ssp245(data_dir):
    """ssp245 targets with all ensemble members, precipitation converted to mm/day."""
    ds = xr.open_dataset(f"{data_dir}/outputs_ssp245.nc")[TARGETS]
    ds["pr"] = ds["pr"] * 86400
    ds["pr90"] = ds["pr90"] * 86400
    return ds


def global_mean(da):
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(["lat", "lon"])


def nrmse(truth, pred):
    """Spatial, global and total NRMSE of one target. truth is the ensemble mean."""
    t, p = truth.sel(time=PERIOD), pred.sel(time=PERIOD)
    assert t.sizes["time"] == p.sizes["time"] == 21, "need every year of 2080-2100"
    norm = abs(float(global_mean(t.mean("time"))))
    weights = np.cos(np.deg2rad(t.lat))
    spatial = float(np.sqrt(((t.mean("time") - p.mean("time")) ** 2).weighted(weights).mean(["lat", "lon"]))) / norm
    glob = float(np.sqrt(((global_mean(t) - global_mean(p)) ** 2).mean("time"))) / norm
    return {"Spatial": spatial, "Global": glob, "Total": spatial + 5 * glob}


def variability_paper(members):
    """The paper's "Variability" row: how much the ssp245 members themselves vary.

    spatial = global mean of the member std of the 2080-2100 mean maps
    global  = std of the yearly global means, over members and years
    Reproduces the published row exactly, so it also checks this module against the paper.
    """
    y = members.sel(time=PERIOD)
    norm = abs(float(global_mean(y.mean("member").mean("time"))))
    spatial = float(global_mean(y.mean("time").std("member"))) / norm
    glob = float(global_mean(y).std()) / norm
    return {"Spatial": spatial, "Global": glob, "Total": spatial + 5 * glob}


def variability_floor(members):
    """Rough best achievable NRMSE given internal variability.

    Scores each member against the mean of the others. If members are independent with equal
    noise, an emulator that is perfect at the forced response scores sqrt(1/3) / sqrt(1 + 1/2)
    (about 0.47) times that against a 3-member mean.
    """
    n = members.sizes["member"]
    scores = [nrmse(members.drop_sel(member=m).mean("member"), members.sel(member=m))
              for m in members.member.values]
    factor = np.sqrt(1 / n) / np.sqrt(1 + 1 / (n - 1))
    floor = {k: float(factor * np.mean([s[k] for s in scores])) for k in ["Spatial", "Global"]}
    floor["Total"] = floor["Spatial"] + 5 * floor["Global"]
    return floor
