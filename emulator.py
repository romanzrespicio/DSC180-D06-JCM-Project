"""A differentiable ClimateBench emulator in JAX (Week 7, Task 1).

A pattern-scaling hybrid, built from Week3_PatternScaling_JAX.ipynb:

    forcing            F_ghg = c_co2 * CO2 + c_ch4 * lag(CH4, tau_ch4)
                       F_aer = so2_pc @ w_so2 + bc_pc @ w_bc
    two-box response   R(F) = (1 - f) lag(F, tau_fast) + f lag(F, tau_slow)       (as in FaIR)
    global-mean core   T_ghg = R(F_ghg),  T_aer = R(F_aer)
    patterns           tas(cell) = a + b_ghg * T_ghg + b_aer * T_aer

Both forcings share one thermal response (a fast and a slow box); CH4 also has its own lag because
the input is emissions, not concentration. CO2 needs no extra lag: its input is already cumulative.
With n_box=1 there is only the fast box (f = 0).

so2_pc and bc_pc are the first N_EOF principal components of the SO2 and BC emission maps, so the
emulator sees the full maps and jax.grad with respect to SO2 is a map. The core (coefficients and
log tau, slow fraction) is fit with Optax on the global-mean tas of the training scenarios; the patterns are a
closed-form least-squares fit per grid cell on [1, T_ghg, T_aer].

All scaling happens inside `emulate`, so it takes the raw ClimateBench inputs and gradients come out
in their units: CO2 cumulative GtCO2, CH4 GtCH4/yr, SO2 and BC kg m-2 s-1.

Inputs are full series from 1850: for an SSP, the historical inputs come first (the lags need the
2014 state). `scenario_inputs` builds them, and the SSP years are the last 86 of the output.
"""
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr

DATA = os.path.expanduser("~/data/climatebench")
TRAIN = ["historical", "hist-GHG", "hist-aer", "ssp126", "ssp370", "ssp585"]
N_EOF = 2          # chosen by leave-one-scenario-out CV over the training scenarios (2, 3, 5 tried; also n_box 1 vs 2)
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


def global_core(core, const, co2, ch4, so2, bc):
    """Global-mean warming split into its GHG and aerosol parts, each (year,)."""
    so2_pc = (so2.reshape(len(so2), -1) @ const["so2_eof"].T) / const["so2_sd"]
    bc_pc = (bc.reshape(len(bc), -1) @ const["bc_eof"].T) / const["bc_sd"]
    tau_ch4, tau_fast, tau_slow = jnp.exp(core["log_tau"])
    f = jax.nn.sigmoid(core["logit_slow"]) if "logit_slow" in core else 0.0

    def response(F):
        return (1 - f) * lag(F, tau_fast) + f * lag(F, tau_slow)

    f_ghg = core["c_co2"] * co2 / const["co2_scale"] + core["c_ch4"] * lag(ch4 / const["ch4_scale"], tau_ch4)
    f_aer = so2_pc @ core["w_so2"] + bc_pc @ core["w_bc"]
    return response(f_ghg), response(f_aer)


def emulate_with(params, co2, ch4, so2, bc):
    """tas (year, lat, lon) for raw inputs from 1850. Pure; differentiable in params and inputs."""
    t_ghg, t_aer = global_core(params["core"], params["const"], co2, ch4, so2, bc)
    p = params["pattern"]                                  # (3, n_cells): a, b_ghg, b_aer
    y = p[0] + jnp.outer(t_ghg, p[1]) + jnp.outer(t_aer, p[2])
    return y.reshape(-1, N_LAT, N_LON)


# ---------------------------------------------------------------- fit

def _eofs(maps, n_eof):
    """First n_eof EOFs (rows) of the maps and the std of their projections.

    No mean is removed, so zero emissions project to zero and give no aerosol warming.
    """
    flat = maps.reshape(len(maps), -1)
    if n_eof == 0:                                   # one basis map: the area-weighted global mean
        eof = np.asarray(area_weights()).reshape(1, -1)
    else:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        eof = vt[:n_eof]
    sd = (flat @ eof.T).std(0)
    return jnp.asarray(eof), jnp.asarray(sd)


def fit(train=TRAIN, n_eof=N_EOF, n_box=1, steps=3000, lr=0.02, verbose=False):
    """Fit the emulator on the training experiments. Uses no data from experiments outside `train`."""
    inputs = {exp: scenario_inputs(exp) for exp in train}
    keep = {exp: n_years(exp) for exp in train}
    tas = {exp: jnp.asarray(load_tas(exp).values.reshape(keep[exp], -1)) for exp in train}
    w = area_weights().ravel()
    gmt = {exp: tas[exp] @ w for exp in train}

    # input scaling and aerosol EOFs from the training inputs (each experiment's own years)
    own = lambda i: np.concatenate([np.asarray(inputs[e][i][-keep[e]:]) for e in train])
    const = {"co2_scale": float(own(0).std()), "ch4_scale": float(own(1).std())}
    for name, i in [("so2", 2), ("bc", 3)]:
        const[f"{name}_eof"], const[f"{name}_sd"] = _eofs(own(i), n_eof)

    def predict_gmt(core, exp):
        t_ghg, t_aer = global_core(core, const, *inputs[exp])
        return (t_ghg + t_aer)[-keep[exp]:]

    def loss(core):
        err = jnp.concatenate([predict_gmt(core, e) - gmt[e] for e in train])
        return jnp.mean(err ** 2)

    core = {"c_co2": 0.5, "c_ch4": 0.1, "w_so2": jnp.zeros(max(n_eof, 1)), "w_bc": jnp.zeros(max(n_eof, 1)),
            "log_tau": jnp.log(jnp.array([10.0, 4.0, 50.0]))}         # tau_ch4, tau_fast, tau_slow
    if n_box == 2:
        core["logit_slow"] = 0.0
    core = jax.tree.map(jnp.asarray, core)
    opt = optax.adam(lr)
    state = opt.init(core)

    @jax.jit
    def step(core, state):
        value, g = jax.value_and_grad(loss)(core)
        updates, state = opt.update(g, state)
        return optax.apply_updates(core, updates), state, value

    for i in range(steps):
        core, state, value = step(core, state)
        if verbose and i % 500 == 0:
            print(f"step {i:5d}  global-mean MSE {float(value):.5f}")

    # patterns: per-cell least squares on [1, T_ghg, T_aer] over all training years
    parts = [global_core(core, const, *inputs[e]) for e in train]
    A = jnp.concatenate([jnp.stack([jnp.ones(keep[e]), g[-keep[e]:], a[-keep[e]:]], 1)
                         for e, (g, a) in zip(train, parts)])
    Y = jnp.concatenate([tas[e] for e in train])
    pattern, *_ = jnp.linalg.lstsq(A, Y)
    return {"core": core, "const": const, "pattern": pattern, "train_loss": float(loss(core))}


def make_emulator(params):
    """The Task 1 signature: emulate(co2, ch4, so2, bc) -> tas (year, lat, lon)."""
    return jax.jit(lambda co2, ch4, so2, bc: emulate_with(params, co2, ch4, so2, bc))
