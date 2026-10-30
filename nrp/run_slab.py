"""Run JEM's Earth-like slab configuration on Week 4's boundary conditions, one
calendar year per call to `run_chunked`, saving CALENDAR-MONTH means.

`jem.main ... coupled_run=long_run` writes 30-day-window means, which drift
against the calendar months. A Q-flux climatology has to be binned by calendar
month, so this uses the Python API: `run_chunked(accumulate=monthly_mean(...))`
accumulates the means inside the scan. The accumulator is not checkpointed, so
each year is its own call, saved to `RUN_DIR/monthly/YYYY.nc` before the next
starts; a killed job loses at most the year it was in, and the same command resumes.

Setup (same for both modes; everything else is jem's `earth-slab`):
- atmosphere terrain: Week 4's `hf://bundles/t63/terrain.nc` (the slab land/ocean
  mask is the atmosphere's terrain mask)
- atmosphere forcing, slab SST, slab land, initial sea ice: `--bc`, the T31 copy of
  Week 4's `forcing_pd.nc` from make_slab_bc.py. CO2 = 420 ppmv as in Week 4,
  unless `--bc` carries a `co2` variable (make_co2_forcing.py).

Modes:
- relax: SST relaxed to the climatology (tau = 30 days, earth-slab's setting).
  Starts from climatology and an isothermal atmosphere.
- qflux: SST free, with the Q-flux from make_qflux.py. With `--init RELAX_RUN_DIR`,
  starts from the relaxed run's final state: every component's carry is copied
  except the ocean's params and forcing, which hold forcing_method and the Q-flux
  array (a plain restore would bring back the relaxed run's zero Q-flux).
  With `--branch QFLUX_RUN_DIR`, starts from a qflux run's final state with the
  CO2 of `--bc` (make_slab_co2.py), for the abrupt-CO2 runs.

The clock restarts at 2000-01-01 after `--init` or `--branch`, so the years of a
branched run count from the branch point.

`--tdland DAYS` sets the land temperature relaxation (earth-slab: 1 day, which makes
land a heat sink under CO2; SPEEDY's default: 40). Use the same value for every run
of a chain: a restored state must have the tdland of the run that restores it.

Usage:
    python run_slab.py relax RUN_DIR --bc BC.nc --end 2011-01-01
    python run_slab.py qflux RUN_DIR --bc BC.nc --qflux QFLUX.nc --init RELAX_RUN_DIR --end 2040-01-01
    python run_slab.py qflux RUN_DIR --bc BC_co2x2.nc --qflux QFLUX.nc --branch QFLUX_RUN_DIR --end 2030-01-01
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from hydra import compose, initialize_config_module

import jem.config  # noqa: F401  (registers the ${jcm_data:} resolvers)
from jem import driver, runners
from jem.accumulate import fold_records, monthly_mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_jem import ATM, ICE, OCN, combine, select  # noqa: E402

START = "2000-01-01"
TERRAIN = "hf://bundles/t63/terrain.nc"


def build_cfg(mode, bc, qflux=None, tdland=None):
    overrides = [
        "+configuration=earth-slab",
        f"atmosphere.run.start_time={START}",
        f"atmosphere.terrain.file={TERRAIN}",
        f"atmosphere.forcing.file={bc}",
        "atmosphere.forcing.align=wrap_year",       # what Week 4's hf:// bundle resolved to
        f"ocean.sst_clim_file={bc}",
        f"land.land_clim_file={bc}",
        f"seaice.ice_clim_file={bc}",
        "coupled_run.log_level=WARNING",
    ]
    if mode == "qflux":
        overrides += ["ocean=slab_qflux", f"ocean.q_flux_file={qflux}"]
    if tdland is not None:
        overrides.append(f"land.params.tdland={tdland * 86400.0}")
    with initialize_config_module(config_module="jem.config", version_base=None):
        return compose(config_name="config", overrides=overrides)


def check_tdland(old, new):
    """The restored land must have the tdland this run was configured with."""
    lnd = [n for n in new.components if hasattr(new.components[n].get("params"), "tdland")]
    assert len(lnd) == 1, lnd
    t_old = float(old.components[lnd[0]]["params"].tdland)
    t_new = float(new.components[lnd[0]]["params"].tdland)
    assert t_old == t_new, f"checkpoint has tdland {t_old / 86400:g} d, this run {t_new / 86400:g} d"


def spliced_start(coupler, relax_run_dir, bc, tdland=None):
    """The relaxed run's final state, with the Q-flux ocean's own params and forcing."""
    relax = runners.build_coupler(build_cfg("relax", bc, tdland=tdland))
    old = relax.load_carry(Path(relax_run_dir) / "checkpoint", check_clock=False)
    new = coupler.initialize()
    check_tdland(old, new)
    comps = dict(old.components)
    ocn = [n for n in comps if "state" in comps[n] and hasattr(comps[n]["state"], "sea_surface_temperature")]
    assert len(ocn) == 1, ocn
    ocn = ocn[0]
    comps[ocn] = {**old.components[ocn], "params": new.components[ocn]["params"],
                  "forcing": new.components[ocn]["forcing"].replace(
                      total_heat_flux=old.components[ocn]["forcing"].total_heat_flux)}
    q = np.asarray(comps[ocn]["forcing"].q_flux)
    logging.warning("spliced start from %s; ocean Q-flux range %.1f .. %.1f W m-2",
                    relax_run_dir, q.min(), q.max())
    return coupler.restart_clock(old.replace(components=comps))


def branched_start(coupler, qflux_run_dir):
    """A qflux run's final state, with this run's CO2.

    The atmosphere's carry holds its ForcingData, co2_vmr included, so a plain
    restore would keep the old run's CO2 (420 ppmv) whatever `--bc` says.
    """
    old = coupler.load_carry(Path(qflux_run_dir) / "checkpoint", check_clock=False)
    new = coupler.initialize()
    check_tdland(old, new)
    comps = dict(old.components)
    atm =[n for n in comps if hasattr(comps[n].get("forcing"), "co2_vmr")]
    assert len(atm) == 1, atm
    atm = atm[0]
    co2_old = float(old.components[atm]["forcing"].co2_vmr)
    co2 = new.components[atm]["forcing"].co2_vmr
    comps[atm] = {**old.components[atm], "forcing": old.components[atm]["forcing"].replace(co2_vmr=co2)}
    logging.warning("branched from %s; CO2 %.1f -> %.1f ppmv", qflux_run_dir, co2_old, float(co2))
    return coupler.restart_clock(old.replace(components=comps))


def to_dataset(coupler, monthly, accumulator, seg_start, seg_end):
    means = monthly.finalize(accumulator)
    _, counts = accumulator
    if isinstance(counts, dict):
        means = {k: fold_records(means[k], counts[k]) for k in means}
    ds = coupler.to_xarray(means)
    # Twelve bins, January first; months outside the segment are NaN and are dropped
    # here, before `combine` reads the ocean mask off frzmlt (NaN != 0).
    months = pd.date_range(f"{seg_start.year}-01-01", periods=12, freq="MS")
    keep = (months >= seg_start) & (months < seg_end)
    ds = {k: v.assign_coords(time=months).isel(time=keep) for k, v in ds.items()}
    names = sorted(ds)
    atm = next(k for k in names if "shortwave_rad.fsol" in ds[k])
    ocn = next(k for k in names if "sea_surface_temperature" in ds[k])
    ice = next(k for k in names if "ice_fraction" in ds[k])
    ocn_names = {**OCN, "forcing_q_flux": "qflux"}
    out = combine(select(ds[atm], atm, ATM), select(ds[ocn], ocn, ocn_names), select(ds[ice], ice, ICE))
    if "qflux" in out:
        out["qflux"] = out.qflux.where(out.ocean_mask == 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["relax", "qflux"])
    p.add_argument("run_dir")
    p.add_argument("--bc", required=True)
    p.add_argument("--qflux")
    p.add_argument("--init", help="relaxed run directory to start a qflux run from")
    p.add_argument("--branch", help="qflux run directory to branch from, with --bc's CO2")
    p.add_argument("--end", required=True, help="end date, e.g. 2040-01-01")
    p.add_argument("--tdland", type=float, help="land relaxation time in days (earth-slab: 1, SPEEDY: 40)")
    a = p.parse_args()
    assert (a.mode == "qflux") == (a.qflux is not None), "--qflux goes with mode qflux only"
    assert not (a.init and a.branch), "--init or --branch, not both"
    assert a.branch is None or a.mode == "qflux", "--branch goes with mode qflux only"
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

    run_dir = Path(a.run_dir).expanduser().resolve()
    (run_dir / "monthly").mkdir(parents=True, exist_ok=True)
    bc = str(Path(a.bc).expanduser().resolve())
    qflux = str(Path(a.qflux).expanduser().resolve()) if a.qflux else None
    cfg = build_cfg(a.mode, bc, qflux, a.tdland)
    coupler = runners.build_coupler(cfg)
    monthly = monthly_mean(coupler)
    ckpt = run_dir / "checkpoint"

    start, end = pd.Timestamp(START), pd.Timestamp(a.end)
    for year in range(start.year, (end - pd.Timedelta(days=1)).year + 1):
        seg_start = max(start, pd.Timestamp(year, 1, 1))
        seg_end = min(end, pd.Timestamp(year + 1, 1, 1))
        out_file = run_dir / "monthly" / f"{seg_start.year}.nc"
        if out_file.exists():
            continue
        init = None
        if a.init and not (ckpt / "carry.msgpack").exists():
            init = spliced_start(coupler, a.init, bc, a.tdland)
        if a.branch and not (ckpt / "carry.msgpack").exists():
            init = branched_start(coupler, a.branch)
        days = (seg_end - seg_start).days
        t0 = time.time()
        res = driver.run_chunked(
            coupler, end_time=seg_end.strftime("%Y-%m-%d"), chunk=f"{days} days",
            initial_carry=init, output_dir=run_dir, health_check=None,
            accumulate=monthly, checkpoint_path=ckpt)
        assert res.completed, f"run stopped before {seg_end}"
        ds = to_dataset(coupler, monthly, res.accumulator, seg_start, seg_end)
        assert int(ds.ocean_mask.sum()) == 3051, "ocean mask is not the Week 4 terrain's"
        assert np.isfinite(ds.sst.where(ds.ocean_mask == 1, 0)).all(), f"non-finite SST in {seg_start.year}"
        ds.attrs.update(mode=a.mode, bc=bc, qflux=str(qflux), terrain=TERRAIN,
                        tdland_days=float(cfg.land.params.tdland) / 86400.0,
                        note="calendar-month means; ocean fields NaN over land; hfds_up is upward "
                             "positive and is F_up - Q in a qflux run")
        ds.to_netcdf(out_file, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
        w = np.cos(np.deg2rad(ds.lat))
        g = lambda v: float(ds[v].mean("time").weighted(w).mean(("lat", "lon")))  # noqa: E731
        logging.warning("%s done in %.0f s: tas %.2f K, sst %.2f K, TOA %+.2f W m-2",
                        seg_start.year, time.time() - t0, g("tas"), g("sst"), g("rtnt"))


if __name__ == "__main__":
    main()
