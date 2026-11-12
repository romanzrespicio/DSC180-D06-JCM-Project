"""A differentiable ClimateBench emulator in JAX (Week 7, Task 1).

Pattern scaling driven by a lagged linear model of global-mean warming, built from Week3_PatternScaling_JAX.ipynb:

    inputs        x = [CO2, CH4, so2_pc, bc_pc] / scale
    global core   T = c0 + c . lag(x, tau)                 lag(x)(t) = a lag(x)(t-1) + (1 - a) x(t), a = exp(-1/tau)
    patterns      tas(cell) = a_cell + b_cell * T

so2_pc and bc_pc project the SO2 and BC emission maps on their first N_EOF EOFs, so the emulator sees the full maps
and jax.grad with respect to SO2 is a map. The core is a closed-form least-squares fit to the global-mean tas of the
training scenarios; the patterns are a per-cell least-squares fit on [1, T] over the same years.

All scaling happens inside `emulate`, so it takes the raw ClimateBench inputs and gradients come out in their units:
CO2 cumulative GtCO2, CH4 GtCH4/yr, SO2 and BC kg m-2 s-1.

Inputs are full series from 1850: for an SSP, the historical inputs come first (the lag needs the 2014 state).
`scenario_inputs` builds them, and the SSP years are the last 86 of the output.

Choices (N_EOF, TAU) were made by holding out each training SSP in turn and scoring its 2080-2100 tas with the
ClimateBench total; ssp245 was scored only once the choice was made. See NOTES.md (2026-11-12) for the comparison,
including the earlier Optax-fitted two-component version, which scored worse.
"""
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import xarray as xr

DATA = os.path.expanduser("~/data/climatebench")
TRAIN = ["historical", "hist-GHG", "hist-aer", "ssp126", "ssp370", "ssp585"]
N_EOF = 1          # aerosol EOFs per species; held-out-SSP CV: 0 (global mean) 0.564, 1 0.541, 2 0.640
TAU = 1.0          # lag (years), as in Week 3; CV is flat between 0.5 and 1 and worse above 2
N_LAT, N_LON = 96, 144


# ---------------------------------------------------------------- data

def _inputs(exp):
    ds = xr.open_dataset(f"{DATA}/inputs_{exp}.nc")
    return {k: ds[k].values.astype(np.float64) for k in ["CO2", "CH4", "SO2", "BC"]}


def scenario_inputs(exp):
    """Raw inputs for one experiment from 1850: (co2, ch4, so2, bc) with historical first for SSPs."""
    x = _inputs(exp)
    if exp.startswith("ssp"):
        h = _inputs("historical")
        x = {k: np.concatenate([h[k], x[k]]) for k in x}
    return tuple(jnp.asarray(x[k]) for k in ["CO2", "CH4", "SO2", "BC"])


def n_years(exp):
    """Number of the experiment's own years (at the end of the scenario_inputs series)."""
    return xr.open_dataset(f"{DATA}/inputs_{exp}.nc").sizes["time"]


def load_tas(exp):
    """Ensemble-mean tas of one experiment, (year, lat, lon)."""
    return xr.open_dataset(f"{DATA}/outputs_{exp}.nc")["tas"].mean("member")


def area_weights():
    lat = xr.open_dataset(f"{DATA}/outputs_historical.nc").lat.values
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(N_LON)
    return jnp.asarray(w / w.sum())


# ---------------------------------------------------------------- model

def lag(x, tau):
    """x~(t) = a x~(t-1) + (1 - a) x(t), a = exp(-1/tau), along axis 0."""
    a = jnp.exp(-1.0 / tau)
    step = lambda prev, xt: (a * prev + (1 - a) * xt,) * 2
    _, out = jax.lax.scan(step, x[0], x[1:])
    return jnp.concatenate([x[:1], out])


def features(const, co2, ch4, so2, bc):
    """Scaled, lagged inputs (year, 2 + 2 * N_EOF)."""
    so2_pc = so2.reshape(len(so2), -1) @ const["so2_eof"].T
    bc_pc = bc.reshape(len(bc), -1) @ const["bc_eof"].T
    x = jnp.column_stack([co2, ch4, so2_pc, bc_pc]) / const["scale"]
    return lag(x, const["tau"])


def global_tas(params, co2, ch4, so2, bc):
    """Global-mean warming (year,)."""
    c = params["core"]
    return c[0] + features(params["const"], co2, ch4, so2, bc) @ c[1:]


def emulate_with(params, co2, ch4, so2, bc):
    """tas (year, lat, lon) for raw inputs from 1850. Pure; differentiable in params and inputs."""
    p = params["pattern"]                                  # (2, n_cells): a, b
    y = p[0] + jnp.outer(global_tas(params, co2, ch4, so2, bc), p[1])
    return y.reshape(-1, N_LAT, N_LON)


# ---------------------------------------------------------------- fit

def _eofs(maps, n_eof):
    """First n_eof EOFs (rows) of the maps, or the area-weighted global mean if n_eof == 0.

    No mean is removed, so zero emissions project to zero.
    """
    flat = maps.reshape(len(maps), -1)
    if n_eof == 0:
        return area_weights().reshape(1, -1)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    eof = vt[:n_eof]
    return jnp.asarray(eof * np.sign(eof.sum(1, keepdims=True)))   # sign: positive emissions -> positive PC


def fit(train=TRAIN, n_eof=N_EOF, tau=TAU):
    """Fit the emulator on the training experiments. Uses no data from experiments outside `train`."""
    inputs = {exp: scenario_inputs(exp) for exp in train}
    keep = {exp: n_years(exp) for exp in train}
    tas = {exp: jnp.asarray(load_tas(exp).values.reshape(keep[exp], -1)) for exp in train}
    w = area_weights().ravel()
    gmt = jnp.concatenate([tas[exp] @ w for exp in train])

    # EOFs and input scaling from the training inputs (each experiment's own years)
    own = lambda i: np.concatenate([np.asarray(inputs[e][i][-keep[e]:]) for e in train])
    const = {"so2_eof": _eofs(own(2), n_eof), "bc_eof": _eofs(own(3), n_eof), "tau": tau,
             "scale": jnp.ones(2 + 2 * max(n_eof, 1))}
    unscaled = jnp.concatenate([features(const, *inputs[e])[-keep[e]:] for e in train])  # lag is linear, so
    const["scale"] = unscaled.std(0)                                                       # rescaling commutes

    # global core: least squares on [1, lagged features]
    X = jnp.concatenate([features(const, *inputs[e])[-keep[e]:] for e in train])
    core, *_ = jnp.linalg.lstsq(jnp.column_stack([jnp.ones(len(X)), X]), gmt)

    # patterns: per-cell least squares on [1, T] with the true training global means
    pattern, *_ = jnp.linalg.lstsq(jnp.column_stack([jnp.ones(len(gmt)), gmt]),
                                   jnp.concatenate([tas[e] for e in train]))
    params = {"core": core, "const": const, "pattern": pattern}
    pred = jnp.concatenate([global_tas(params, *inputs[e])[-keep[e]:] for e in train])
    params["train_rmse"] = float(jnp.sqrt(jnp.mean((pred - gmt) ** 2)))
    return params


def make_emulator(params):
    """The Task 1 signature: emulate(co2, ch4, so2, bc) -> tas (year, lat, lon)."""
    return jax.jit(lambda co2, ch4, so2, bc: emulate_with(params, co2, ch4, so2, bc))
