"""Week 4 Task 2: SPEEDY T31 throughput, compile time apart from run time.

Model._run_from_state is jitted with the model and the run length as static
arguments, so call 1 (`run`) compiles and runs, and call 2 (`resume`, same
length) only runs. compile ~= call 1 - call 2.

Usage:
    python throughput.py [--days 365]              # default device (GPU if visible)
    JAX_PLATFORMS=cpu python throughput.py         # force CPU
"""
import argparse
import json
import platform
import time
from importlib import resources

import jax

from jcm.forcing import ForcingData
from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData

p = argparse.ArgumentParser()
p.add_argument("--days", type=float, default=365.0)
args = p.parse_args()

# same grid and time step as +configuration=speedy-t31; the packaged T30
# climatology (as in 01_jcm_demo) avoids a download, and the cost per step
# does not depend on the forcing values
data_dir = resources.files("jcm.data.bc.t30.clim")
coords = get_speedy_coords()
terrain = TerrainData.from_file(data_dir / "terrain.nc", coords=coords)
forcing = ForcingData.from_file(data_dir / "forcing.nc", coords=coords, align_mode="wrap_year")
model = Model(coords=coords, terrain=terrain, time_step=15)

# save_interval must divide total_time exactly; 5 days divides 365
kw = dict(forcing=forcing, save_interval=5.0, total_time=args.days, output_averages=True)

t0 = time.perf_counter()
jax.block_until_ready(model.run(**kw))
call1 = time.perf_counter() - t0

t0 = time.perf_counter()
jax.block_until_ready(model.resume(**kw))
call2 = time.perf_counter() - t0

years = args.days / 365.0
print(json.dumps({
    "device": str(jax.devices()[0]),
    "device_kind": jax.devices()[0].device_kind,
    "host": platform.node(),
    "sim_days": args.days,
    "call1_compile_plus_run_s": round(call1, 1),
    "call2_run_s": round(call2, 1),
    "compile_s": round(call1 - call2, 1),
    "sim_years_per_hour": round(years / (call2 / 3600), 1),
}))
