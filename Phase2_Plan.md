# Phase II plan: learning the ocean in JEM

Working plan for the Quarter 2 project proposal. The proposal (written proposal, schedule, elevator pitch) is due
Friday, December 4. Numbers marked "estimate" are not measured yet.

## Contents

1. [Summary](#1-summary)
2. [The problem in simple words](#2-the-problem-in-simple-words)
3. [Background: why the eastern Pacific is unusual](#3-background-why-the-eastern-pacific-is-unusual)
4. [What Phase I already shows](#4-what-phase-i-already-shows)
5. [The model](#5-the-model)
6. [Data](#6-data)
7. [Experiments](#7-experiments)
8. [How we measure success](#8-how-we-measure-success)
9. [Compute cost](#9-compute-cost)
10. [Team split](#10-team-split)
11. [Schedule (6–7 weeks)](#11-schedule-67-weeks)
12. [Risks and fallbacks](#12-risks-and-fallbacks)
13. [Preliminary analysis before December 4](#13-preliminary-analysis-before-december-4)
14. [Proposal parts](#14-proposal-parts)
15. [Elevator pitch](#15-elevator-pitch)
16. [Related work to read and cite](#16-related-work-to-read-and-cite)
17. [Questions for Duncan](#17-questions-for-duncan)

---

## 1. Summary

**Title:** Learning the ocean to fix climate projections for the eastern Pacific and western US.

**Question:** Simple climate models represent the ocean as a single layer with a fixed heat-transport correction, which
cannot change as the climate warms. Can a neural network, trained by gradient descent inside a differentiable climate
model, learn how ocean heat transport responds to warming? Does this change the projected warming patterns, especially
over the eastern Pacific and western North America?

**Motivation:** Upwelling and ocean currents control the eastern Pacific, and simple ocean models cannot represent them.
The temperature pattern there affects global climate sensitivity, West Coast temperatures, the marine layer, and the
effect of El Niño on California rain. Hybrid physics–ML models keep simple models fast but make them more realistic. A
differentiable model lets us train the ocean part directly against the result that we want it to get right.

**What is new:** a learned, state-dependent ocean heat-flux term, trained online inside a differentiable coupled model,
and a direct comparison of online and offline training. The novelty claim needs a literature check (Section 16).

This is the "Learning the ocean" idea from Duncan's Phase II list, with a regional focus.

## 2. The problem in simple words

**The problem**

1. Scientists use computer programs to calculate how the Earth will warm.
2. Our program (JEM) is small and fast. But it treats the ocean as a flat pool of water that does not move.
3. The real ocean has currents that carry heat from one place to another. Deep water also takes heat down from the
   surface.
4. In our program, the currents are frozen at present-day values. They cannot change when the Earth warms.
5. Because of this, the program gets some places wrong. The largest error is in the Pacific Ocean next to South America
   and the US West Coast. Our program warms that area too much.

**Our idea**

1. We add a small neural network to the ocean part of the program.
2. The network learns how the currents change when the Earth warms.
3. It learns from a big, slow climate model (NorESM2) that has a real moving ocean.
4. Our program can calculate its own gradients. Thus the network can learn while the program runs, not only from a
   table of numbers. This "learn while it runs" method is the new part.

**How we check it**

1. We run our program with the old frozen ocean and with the new learned ocean.
2. We compare both with NorESM2 and with real ocean measurements (satellite sea temperatures).
3. If the learned ocean gets the Pacific warming closer to NorESM2, the idea works.
4. We also check that the network does better than a very simple formula. If it does not, a simple formula is enough,
   and that is also a useful result.

**One limit:** JEM holds the land temperature close to present-day values. Thus it can show the ocean change well, but
its results for West Coast land are less reliable.

## 3. Background: why the eastern Pacific is unusual

The "eastern Pacific" in this project has two regions. Both are cold because cold water comes up from below, but only
the first one controls El Niño.

1. **The equatorial east Pacific:** the ocean along the equator, next to Ecuador and Peru. It controls El Niño.
2. **The California coast:** the ocean along the US West Coast. It controls the coastal fog (the marine layer).

### Why the equatorial east Pacific is cold

The east Pacific is about 5–8 °C colder than the west Pacific, although the Sun heats both strongly. Three things cause
this:

1. **The trade winds push the warm water west.** Along the equator, the winds blow from east to west. A thick layer of
   warm water (the "warm pool") collects near Indonesia and Australia.
2. **Cold water comes up in the east (upwelling).** The winds remove the surface water, and the Earth's rotation turns
   it away from the equator to the north and to the south. Cold water from below moves up to replace it.
3. **The cold layer is close to the surface in the east.** The boundary between warm surface water and cold deep water
   (the thermocline) is about 150–200 m deep in the west and about 50 m deep in the east. Thus upwelling in the east
   brings up water that is really cold.

```
 West (Indonesia)                                East (Peru)
   warm, wet          <-- trade winds blow west --
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  surface
  warm pool, thick                       cold water comes up ^
 ..............
               ..........
 cold deep water         ............ thermocline .....
```

The result is a band of cold water along the equator in the east, the "cold tongue".

### The loop that keeps it in place (Bjerknes feedback)

1. The warm west makes the air rise and rain. The cold east makes the air sink and stay dry.
2. This makes a large air circulation along the equator (the Walker circulation). Near the surface, the air moves from
   east to west: these are the trade winds.
3. The trade winds push warm water west and cause upwelling in the east, which keeps the east cold.
4. A cold east and a warm west make the trade winds stronger.

This loop can increase a small change until it becomes a large change.

### How El Niño happens

1. The trade winds become weaker.
2. The warm water in the west flows back east.
3. In the east, the thermocline becomes deeper. Upwelling now brings up warm water, not cold water.
4. The east becomes warmer, so the west–east difference becomes smaller.
5. A smaller difference makes the trade winds weaker, so the change becomes larger.

After about 6–18 months, ocean waves move the thermocline back, and the process stops. Often the system then goes too
far in the other direction (stronger winds, a colder east): that is La Niña. The cycle repeats every 2–7 years.

**Effect on California:** in El Niño, the tropical rain moves east into the central Pacific. Its heat changes the path of
the North Pacific jet stream. In many El Niño winters, storms move toward Southern California, so it is often wetter
than normal. This does not always occur: the strong El Niño of 2015–16 did not bring a wet winter to Southern
California.

### The California coast

Along the West Coast, the winds blow from north to south, parallel to the coast. The Earth's rotation turns the surface
water to the right of the wind, which is away from the coast. Cold water comes up near the coast to replace it. This
cold water cools the air directly above it, warmer air stays above the cool layer, and moisture stays trapped near the
surface. The result is the low clouds of "May Gray" and "June Gloom".

### Why this matters for the project

- **A slab ocean cannot do any of this.** It has no currents, no upwelling, and no thermocline. The Q-flux only adds a
  fixed cooling in each place. In Week 5, the Q-flux removed more than 100 W m⁻² (watts per square meter) in the east
  Pacific, about 25 times the forcing from doubled CO2 (about 4 W m⁻²).
- **A warmer climate can change the upwelling.** If the winds or the thermocline change, the cooling from upwelling
  changes too. A fixed Q-flux cannot show this.
- **A plausible reason for the Week 5 result.** The "ocean thermostat" theory says that upwelling of cold, deep water
  slows the warming of the east Pacific. A fixed Q-flux has no thermostat. That is consistent with JEM's east Pacific
  warming of 1.57 K per K, against 1.09 in NorESM2. The project can test this explanation. Week 5 does not prove it.
- **Scientists do not agree on the future.** Most climate models warm the east Pacific faster than the west. But from
  about 1980 to 2020, the observed east Pacific became slightly cooler, which the models did not reproduce. OISST lets
  us show the observed trend.
- **The pattern effect.** Cold water in the east Pacific keeps large areas of low clouds, which reflect sunlight. If the
  east warms, the clouds can decrease and the planet warms more. This links the regional question to climate
  sensitivity.

**Resolution limit:** JEM's atmosphere is SPEEDY at T31. Each grid cell is about 400 km wide. The California coastal
upwelling zone is only about 50–100 km wide. The model can show the equatorial cold tongue, but not the narrow coastal
upwelling or the marine layer. We do not promise marine-layer results.

## 4. What Phase I already shows

From [Week5_WarmingPatterns.ipynb](notebooks/Week5_WarmingPatterns.ipynb) (warming per K of global warming, 2xCO2):

| Region | NorESM2-LM | JEM slab (Week 5) |
|---|---|---|
| East equatorial Pacific | 1.09 | 1.57 |
| West equatorial Pacific | 0.90 | 0.97 |
| Subpolar North Atlantic | 0.40 | about 1.0 |
| Southern Ocean | 0.29 | about 1.0 |
| Arctic (60–90N) | 2.06 | 0.66 |
| Land/ocean warming ratio | 1.78 | 0.93 |

The notebook says that a Q-flux slab "keeps ocean heat transport fixed and has no deep ocean", so it cannot produce the
slow-warming regions.

From Week 5 Task 2 (Q-flux control, README):

- The Q-flux is below −100 W m⁻² in the east Pacific. The ocean mean is −13.4 W m⁻².
- The README calls it "a flux correction for SPEEDY's surface-flux biases, not a realistic ocean heat transport".
- The equatorial cold tongues are too warm by +1 to +2 K.

From Week 5 Tasks 3–5 ([Week5_CO2Runs.ipynb](notebooks/Week5_CO2Runs.ipynb)):

- The slab reaches 90% of its final warming by year 4. NorESM2 takes decades.
- With `tdland = 1 day`, land surface warms only 0.35 K at 2xCO2. With `tdland = 10 days` (Casper side note), it
  warms 1.84 K. 20 and 40 days blew up.

## 5. The model

The Week 5 slab equation for each ocean cell:

$$C \frac{dT}{dt} = F_\text{net} + Q_0(x, \text{month})$$

$Q_0$ is the Week 5 Q-flux and stays fixed. The project adds a learned term:

$$C \frac{dT}{dt} = F_\text{net} + Q_0(x, \text{month}) + \Delta Q_\theta(s)$$

- **Inputs $s$ for each column:** the SST anomaly from the control, the global-mean warming, the surface wind stress
  (upwelling depends on the winds), latitude and longitude.
- **Network:** a small MLP applied to each column (about 6 inputs, 2 hidden layers of 64). Option: a small CNN for
  neighbor information.
- **Two parts:** $\Delta Q_\theta = \Delta Q_\text{transport} + \Delta Q_\text{uptake}$.
  - $\Delta Q_\text{transport}$: the change in ocean heat transport. Its global mean is removed (transport moves heat,
    it does not create it).
  - $\Delta Q_\text{uptake}$: heat that goes into the deep ocean. Its global mean is negative. The slab has no deep
    ocean, which is why it warms too fast.
- **Why $Q_0$ stays separate:** $Q_0$ corrects SPEEDY's present-day biases. If the network learned all of $Q$, it would
  mix the bias correction with the warming response.

## 6. Data

| Data | Variables | Role | Status |
|---|---|---|---|
| NorESM2-LM, CMIP6 (Pangeo or ESGF) | `tos`, `hfds`, `tauu`, `tauv`; piControl, abrupt-4xCO2, 1pctCO2 | Training target: how ocean heat transport responds to warming | New: get before Dec 4 |
| ERA5 | Surface fluxes, wind stress | Evaluate the present-day control; set the input scales | Have it (`week4/get_obs.py`) |
| NOAA OISST | Observed SST, 1982 to now | Evaluate the present-day east Pacific bias; stretch: observed trend | Public: get before Dec 4 |
| Phase I JEM runs | Q-flux control, 2x and 4xCO2, `tdland` 1 and 10 days | Baseline (fixed Q-flux) | Done (`results/w5-*`) |

Observed data (ERA5, OISST) gives only the present-day climate. It cannot teach how the ocean responds to warming. Only
NorESM2 can do that.

**Training target from NorESM2.** For a slab with the same heat capacity $C$, NorESM2 gives an implied Q:

$$Q_\text{implied} = C \frac{dT_\text{NorESM}}{dt} - \text{hfds}$$

Calculate it for piControl and abrupt-4xCO2. Then
$\Delta Q_\text{target} = Q_\text{implied}(\text{4xCO2}) - Q_\text{implied}(\text{piControl})$. Regrid to T31
(48 × 96). Use the piControl variability to find the noise in the target.

## 7. Experiments

| Run | Ocean | Purpose |
|---|---|---|
| E0 | Week 5 fixed Q-flux | Baseline (done) |
| E1 | $\Delta Q = -\kappa \, \Delta T$, one parameter, fitted by gradient | Physical baseline: the network must beat this |
| E2 | $\Delta Q$ = pattern × global $\Delta T$ | Pattern-scaling baseline |
| E3 | $\Delta Q_\theta$ trained offline on NorESM2 states | ML, offline |
| E4 | $\Delta Q_\theta$ trained online inside JEM | ML, online (main new result) |

E1 and E2 are important. Without them, a reviewer can say that a one-parameter deep-ocean term gives the same
improvement as the network.

**Offline (E3).** Fit $\Delta Q_\theta$ to (NorESM2 state, $\Delta Q_\text{target}$) pairs, then put it in JEM. The JEM
atmosphere differs from NorESM2's, so the inputs move out of the training range. The run can drift or blow up. This is
the known offline problem.

**Online (E4).** Run JEM with $\Delta Q_\theta$ and minimize the difference between the JEM SST response and the NorESM2
SST response. Use short windows (about 1 month) and `jax.checkpoint`. The gradient of SST with respect to Q goes mostly
through the slab equation, so it can be better behaved than the atmospheric gradients in Week 6 (reliable for about 4
days). But the atmospheric feedbacks are chaotic, so test this first (Section 13).

**Train and test split.** Train on abrupt-4xCO2. Test on 1pctCO2 (a different rate of forcing) and 2xCO2 (a different
size of forcing).

**Base setup.** Use `tdland = 10 days`, so that land can warm (Section 4).

## 8. How we measure success

| Metric | NorESM2 target | JEM now (Week 5) |
|---|---|---|
| East Pacific warming per K, east / west | 1.09 / 0.90 | 1.57 / 0.97 |
| Subpolar North Atlantic warming per K | 0.40 | about 1.0 |
| Southern Ocean warming per K | 0.29 | about 1.0 |
| Time to 90% of the warming in abrupt CO2 | Decades | 4 years |
| Pattern correlation of $\Delta$SST/$\Delta T$ with NorESM2 | 1 | Calculate in week 1 |

Secondary metrics:

- Western North America tas and pr (with the land caveat).
- The Gregory slope (the pattern effect on climate sensitivity).
- The present-day cold-tongue bias against OISST.

**Hypothesis (draft):** the online $\Delta Q_\theta$ reduces the east Pacific east–west warming error by at least 50%
compared with E0, and it beats E1 on the 1pctCO2 test. Change the numbers after the preliminary analysis.

## 9. Compute cost

### Running the model with the network: under 1% slower (estimate)

- JEM updates the slab ocean once per day (`coupling: daily`). The atmosphere takes a step every 30 minutes, so it runs
  48 steps for each ocean update.
- The MLP on 3,051 ocean cells needs about 30 million operations for each call. The atmosphere does about 1,000 times
  more work in one day.
- On a GPU, JAX compiles the network into the same program as the model.

Measure it in week 1: run 1 year with a zero network and 1 year without it.

| Hardware | Measured speed |
|---|---|
| NRP GPU (GTX 1080 Ti), JCM atmosphere only (Week 4) | 55.8 model years per hour |
| Laptop CPU (M2), JEM with slab (Week 5) | about 8 model years per hour |

### Training the network

| Method | What it needs | Time (estimate) |
|---|---|---|
| Offline (E3) | Fit to a table of NorESM2 numbers; the climate model does not run | Minutes |
| Online (E4) | Forward run plus backward gradient for each training step | Hours |

Each online step costs about 3–4 forward runs (the backward pass costs 2–3 times the forward run, and
`jax.checkpoint` runs parts of the model again). On the NRP GPU:

1. A 1-month window takes about 5 seconds forward.
2. One training step takes about 20 seconds.
3. 1,000 training steps take about 5–6 hours.

Limits: the GPU has 11 GB of memory, and the JupyterHub server has 8 GB RAM and 1 CPU core. Use NRP Kubernetes Jobs for
long runs. If stable gradients need windows longer than 1 month, each step costs more.

## 10. Team split

| Member | Area | Main output |
|---|---|---|
| A | Data: NorESM2 download, regrid to T31, $\Delta Q_\text{target}$, OISST and ERA5 checks | Training dataset and evaluation targets |
| B | Offline: train $\Delta Q_\theta$, run E1–E3 in JEM | Offline results and baselines |
| C | Online: training loop through JEM, gradient checks, `jax.checkpoint` | E4 and the offline/online comparison |
| D | Evaluation: metrics, maps, western US analysis, figures | Results section and website |

Annaly built the Q-flux control (Week 5 Task 2), so Annaly is a natural choice for B or C. Assign the others by
strength.

## 11. Schedule (6–7 weeks)

| Week | A (data) | B (offline) | C (online) | D (evaluation) |
|---|---|---|---|---|
| 1 | Regrid NorESM2 `tos`, `hfds`, wind stress to T31 | Put a zero $\Delta Q_\theta$ in `run_slab.py`; check that E0 is unchanged; time it | Gradient of 1-month SST with respect to Q; compare with finite difference | Metric code; score E0 against NorESM2 |
| 2 | Calculate $\Delta Q_\text{target}$ for 4xCO2 and its noise from piControl | Fit E1 ($\kappa$) and E2 (pattern); run 30-year 4xCO2 | Find the longest window that gives stable gradients | OISST and ERA5 present-day evaluation of E0 |
| 3 | Make the 1pctCO2 test targets | Train offline $\Delta Q_\theta$ (E3) | First online training on 1-month windows | Score E1 and E2 |
| 4 | Input scaling, extra input variables | Run E3 for 4xCO2 and 1pctCO2; check drift | Online training at full scale (E4) | Score E3; east Pacific maps |
| 5 | Stretch data (OISST trend or Veros) | Ablation: remove wind stress from the inputs | Run E4 for 4xCO2 and 1pctCO2 | Score E4; offline/online comparison |
| 6 | Report: data section | Report: offline section | Report: online section | Western US analysis; website |
| 7 | Buffer and polish | Buffer and polish | Buffer and polish | Buffer and polish |

The course asks for about 11 hours per person per week on the project.

## 12. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Online gradients are unstable over months | Shorter windows and SST tendency matching; offline (E3) is still a complete result |
| Offline $\Delta Q_\theta$ blows up in JEM | That is a result: it shows why online training is needed |
| The network does not beat E1 | That is a result: the missing physics is deep uptake, not transport |
| NorESM2 variables are not on Pangeo | Use ESGF |
| Western US land results are poor | Report them with the `tdland` caveat; the ocean results are the main result |
| The novelty claim is wrong | Ask Duncan now and change the framing |

Each fallback still gives a result.

## 13. Preliminary analysis before December 4

The course needs proof that we can get the new data, that it contains the needed information, that it has enough
quality and quantity, and that we have the compute.

1. Discuss this plan with Duncan.
2. Download NorESM2-LM `tos` and `hfds` for piControl and abrupt-4xCO2.
3. Calculate $\Delta Q_\text{target}$ in the east Pacific. Show that it is larger than the piControl noise.
4. Download OISST. Calculate the east Pacific climatology.
5. Take `jax.grad` of a 1-month east Pacific SST mean with respect to Q in JEM. Compare it with a finite difference.
6. Record the time and memory of step 5 on the NRP GPU.

Steps 3 and 5 are the most important. Step 3 shows that the target exists. Step 5 shows that online training is
possible.

## 14. Proposal parts

Use the course LaTeX template on Overleaf.

1. **Context (general audience):** simple climate models keep the ocean frozen in its present state. We teach the ocean
   to change as the climate warms, and we test whether this changes the projected warming off the West Coast.
2. **Technical problem statement (expert):** the equations in Section 5, the E0–E4 matrix in Section 7, the metrics in
   Section 8, and the related work in Section 16.
3. **Primary output:** a report, plus a website with interactive warming maps for E0–E4.
4. **Feasibility:** the results of Section 13.

The schedule (Section 11) is a separate PDF.

## 15. Elevator pitch bullet points

- We need to know how the climate will change in each region to help plan for future changes and help people connect global warming to real effects in their lives
- California’s water resources, fire risk, and coastal fog depend on the eastern Pacific Ocean.  
- Winds push the warm surface water away (west along the equator, offshore along the California coast), and colder deep water comes up to replace it. That cold water cools the West Coast and affects rain, as well as El Niño
- Big climate models simulate ocean currents well, but take weeks for a single run. Our mentor’s lab works on small, faster models that can test many future scenarios. The fast model that we use treats the ocean as a flat slab of water: the currents that move heat around the world are fixed and do not change when the planet warms.
- This makes the Eastern Pacific Ocean warm about 40% faster than in a full model, which affects warming predictions for the West Coast.
- Can we keep a climate model fast but teach the ocean model to respond to warming?
- We want to add a small neural network that learns how ocean heat transport responds to warming. Since our model is differentiable, we can calculate how each input change affects each output. Therefore we can train our network inside the running model against the expected result.
- If successful, we expect our network to cut the error in eastern pacific warming pattern by around half. If this works, a fast model can give more accurate regional answers, especially around the West Coast, closer to the full slower climate model.

The 40% comes from Week 5: JEM warms the east Pacific at 1.57 K per K, and NorESM2 at 1.09.

## 16. Related work to read and cite

Check each one before the proposal. The proposal must state how this project differs from them.

- Kochkov et al. (2024), NeuralGCM: learned physics trained online through unrolled model steps.
- The Ai2 climate modeling group: ML corrections in FV3GFS, and ACE2-SOM (an ML atmosphere coupled to a slab ocean).
- Frezat et al. (2022): offline ("a priori") and online ("a posteriori") training of closures.
- Clement et al. (1996): the ocean dynamical thermostat.
- Seager et al. (2019): the observed east Pacific trend that models do not reproduce.
- Literature on the pattern effect and on slab-ocean Q-flux experiments.

## 17. Questions for Duncan

1. Is a learned, state-dependent Q-flux trained online in JEM new, or do you know of similar work?
2. Is NorESM2 implied Q (from `tos` and `hfds`) a good training target, or is there a better one?
3. Is `tdland = 10 days` the best land setup, or is there a more stable fix?
4. Is the comparison with JEM's Veros coupling realistic as a stretch goal?
5. What compute should we use for long runs: NRP Kubernetes Jobs, or Casper?
