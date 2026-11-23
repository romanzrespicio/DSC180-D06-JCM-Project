# Phase II plan: learning the ocean in JEM

Working plan for the Quarter 2 project proposal. The proposal (written proposal, schedule, elevator pitch) is due Friday, December 4. Numbers marked "estimate" are not measured yet.


## Contents

1. Summary
2. The problem in simple words
3. Background: why the eastern Pacific is unusual
4. What Phase I already shows
5. The model
6. Data
7. Experiments
8. Training modes and loss
9. How we measure success
10. Compute cost
11. Team split
12. Schedule (7 weeks)
13. Risks and fallbacks
14. Preliminary analysis before December 4
15. Proposal parts
16. Elevator pitch and communication
17. Related work to read and cite
18. Questions for Duncan

---

## 1. Summary

**Working title:** Learning the ocean: fixing a fast climate model's warming pattern in the eastern Pacific, with implications for the western US.
(Possible project name: *Current Events*.)

**Question:** Simple climate models represent the ocean as one layer with a fixed heat-transport correction. It cannot change as the climate warms, and it has no deep ocean to slow the warming. Can a neural network, trained by gradient descent inside a differentiable climate model, learn how ocean heat transport and deep-ocean heat uptake respond to warming? Does online training through the model beat standard offline training? Does the learned ocean reproduce a full climate model's warming pattern, especially over the eastern Pacific?

**Motivation:** Upwelling and ocean currents control the eastern Pacific, and simple ocean models cannot represent them. The temperature pattern there affects global climate sensitivity, West Coast temperatures, the marine layer, and the effect of El Niño on California rain. It is also a region where climate models and observations disagree. Hybrid physics–ML models keep simple models fast but make them more realistic. ML parts trained offline often drift or become unstable when they are coupled to a climate model. Online training fixes this, but it needs a differentiable model, and JEM is differentiable.

**What is new:** a learned, state-dependent ocean (deep-layer exchange plus heat-transport change), trained inside a differentiable coupled model. We directly compare offline, partially online and fully online training. The novelty claim needs a literature check (Section 17).

This is the "Learning the ocean" idea from Duncan's Phase II list, with a regional focus.

---

## 2. The problem in simple words

**The problem**

- Scientists use computer programs to calculate how the Earth will warm.
- Our program (JEM) is small and fast. But it treats the ocean as a thin, flat pool of water that does not move.
- The real ocean has currents that carry heat from one place to another. The deep ocean also takes heat down from the surface and stores it for decades, like a slow battery.
- In our program, the currents are frozen at present-day values, and there is no deep ocean. Nothing can change when the Earth warms.
- Because of this, the program gets some places wrong. The largest error is in the Pacific Ocean next to South America. Our program warms that area too much, and it warms everywhere too quickly (in about 4 years instead of decades).

**Our idea**

- We give the ocean a second, deep layer that warms slowly.
- We add a small neural network. It learns how much heat moves between the surface and the deep layer, and how the currents change when the Earth warms.
- It learns from a big, slow climate model (NorESM2) that has a real moving ocean.
- Our program can calculate its own gradients. Thus the network can learn while the program runs, not only from a table of numbers. We test whether this "learn while it runs" method works better than the usual way.

**How we check it**

- We run our program with the old frozen ocean, with simple physical fixes, and with the learned ocean.
- We compare all of them with NorESM2, and the present-day climate with real ocean measurements (satellite sea temperatures).
- If the learned ocean gets the Pacific warming closer to NorESM2 than the simple fixes do, the idea works. If it does not, a simple fix is enough, and that is also a useful result.
- One limit: JEM holds the land temperature close to present-day values. Thus it shows the ocean change well, but its results for West Coast land are less reliable.

---

## 3. Background: why the eastern Pacific is unusual

The "eastern Pacific" in this project has two regions. Both are cold because cold water comes up from below, but only the first one controls El Niño.

- **The equatorial east Pacific:** the ocean along the equator, next to Ecuador and Peru. It controls El Niño.
- **The California coast:** the ocean along the US West Coast. It controls the coastal fog (the marine layer).

### Why the equatorial east Pacific is cold

The east Pacific is about 5–8 °C colder than the west Pacific, although the Sun heats both strongly. Three things cause this:

1. **The trade winds push the warm water west.** Along the equator, the winds blow from east to west. A thick layer of warm water (the "warm pool") collects near Indonesia and Australia.
2. **Cold water comes up in the east (upwelling).** The winds remove the surface water, and the Earth's rotation turns it away from the equator to the north and to the south. Cold water from below moves up to replace it.
3. **The cold layer is close to the surface in the east.** The boundary between warm surface water and cold deep water (the thermocline) is about 150–200 m deep in the west and about 50 m deep in the east. Thus upwelling in the east brings up water that is really cold.

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

- The warm west makes the air rise and rain. The cold east makes the air sink and stay dry.
- This makes a large air circulation along the equator (the Walker circulation). Near the surface, the air moves from east to west: these are the trade winds.
- The trade winds push warm water west and cause upwelling in the east, which keeps the east cold.
- A cold east and a warm west make the trade winds stronger.
- This loop can increase a small change until it becomes a large change.

### How El Niño happens

1. The trade winds become weaker.
2. The warm water in the west flows back east.
3. In the east, the thermocline becomes deeper. Upwelling now brings up warm water, not cold water.
4. The east becomes warmer, so the west–east difference becomes smaller.
5. A smaller difference makes the trade winds weaker, so the change becomes larger.
6. After about 6–18 months, ocean waves move the thermocline back, and the process stops. Often the system then goes too far in the other direction (stronger winds, a colder east): that is La Niña. The cycle repeats every 2–7 years.

**Effect on California:** in El Niño, the tropical rain moves east into the central Pacific. Its heat changes the path of the North Pacific jet stream. In many El Niño winters, storms move toward Southern California, so it is often wetter than normal. This does not always occur: the strong El Niño of 2015–16 did not bring a wet winter to Southern California.

### The California coast

Along the West Coast, the winds blow from north to south, parallel to the coast. The Earth's rotation turns the surface water to the right of the wind, which is away from the coast. Cold water comes up near the coast to replace it. This cold water cools the air directly above it, warmer air stays above the cool layer, and moisture stays trapped near the surface. The result is the low clouds of "May Gray" and "June Gloom".

### Why this matters for the project

- **A slab ocean cannot do any of this.** It has no currents, no upwelling, no thermocline and no deep ocean. The Q-flux only adds a fixed cooling in each place. In Week 5, the Q-flux removed more than 100 W m⁻² in the east Pacific, about 25 times the forcing from doubled CO2 (about 4 W m⁻²).
- **A warmer climate can change the upwelling.** If the winds or the thermocline change, the cooling from upwelling changes too. A fixed Q-flux cannot show this.
- **The ocean thermostat needs memory.** The "ocean thermostat" theory (Clement et al. 1996) says that upwelling of cold, deep water slows the warming of the east Pacific. This works only because the upwelled water comes from below and has not warmed yet. A slab has no water below it. This is consistent with JEM's east Pacific warming of 1.57 K per K, against 1.09 in NorESM2. The deep layer in Section 5 gives the model this memory. The project can test this explanation; Week 5 does not prove it.
- **Scientists do not agree on the future.** Most climate models warm the east Pacific faster than the west. But from about 1980 to 2020, the observed east Pacific became slightly cooler, which the models did not reproduce (Seager et al. 2019). OISST lets us show the observed trend. **We do not claim to solve this disagreement:** we train against NorESM2, so we can reproduce NorESM2's pattern, not explain the observations.
- **The pattern effect.** Cold water in the east Pacific keeps large areas of low clouds, which reflect sunlight. If the east warms, the clouds can decrease and the planet warms more. This links the regional question to climate sensitivity.
- **Resolution limit.** JEM's atmosphere is SPEEDY at T31. Each grid cell is about 400 km wide. The California coastal upwelling zone is only about 50–100 km wide. The model can show the equatorial cold tongue, but not the narrow coastal upwelling or the marine layer. We do not promise marine-layer results.

---

## 4. What Phase I already shows

From `Week5_WarmingPatterns.ipynb` (warming per K of global warming, 2xCO2):

| Region | NorESM2-LM | JEM slab (Week 5) |
|---|---|---|
| East equatorial Pacific | 1.09 | 1.57 |
| West equatorial Pacific | 0.90 | 0.97 |
| Subpolar North Atlantic | 0.40 | about 1.0 |
| Southern Ocean | 0.29 | about 1.0 |
| Arctic (60–90N) | 2.06 | 0.66 |
| Land/ocean warming ratio | 1.78 | 0.93 |

The notebook says that a Q-flux slab "keeps ocean heat transport fixed and has no deep ocean", so it cannot produce the slow-warming regions. The Subpolar North Atlantic and the Southern Ocean are the main regions of deep-ocean heat uptake, so the deep layer should help most there.

The Arctic and land values mainly come from sea ice and the land setup (`tdland`), not from the ocean. **We do not use them as training targets** (Section 9).

From Week 5 Task 2 (Q-flux control, README):

- The Q-flux is below −100 W m⁻² in the east Pacific. The ocean mean is −13.4 W m⁻².
- The README calls it "a flux correction for SPEEDY's surface-flux biases, not a realistic ocean heat transport".
- The equatorial cold tongues are too warm by +1 to +2 K.

From Week 5 Tasks 3–5 (`Week5_CO2Runs.ipynb`):

- The slab reaches 90% of its final warming by year 4. NorESM2 takes decades.
- With `tdland` = 1 day, land surface warms only 0.35 K at 2xCO2. With `tdland` = 10 days (Casper side note), it warms 1.84 K. 20 and 40 days blew up.

From Week 6: gradients through the atmosphere are reliable for about 4 days.

---

## 5. The model

### Week 5 slab (baseline)

For each ocean cell:

$$C \frac{dT}{dt} = F_{net} + Q_0(x, \text{month})$$

$Q_0$ is the Week 5 Q-flux and stays fixed.

### Why a learned term alone is not enough

If we only add a term that depends on the current state, $\Delta Q_\theta(s)$, the model still has no memory. For example, $\Delta Q = -\kappa\,\Delta T$ gives

$$C \frac{d\,\Delta T}{dt} = F - (\lambda + \kappa)\,\Delta T,$$

so the response time $C/(\lambda + \kappa)$ becomes *shorter*, not longer. The term also never turns off, so it permanently lowers the final warming. Real deep-ocean uptake becomes smaller as the deep ocean warms. Any function of the current state behaves the same way. It can move warming from place to place, but it cannot make warming slow. It also cannot tell abrupt-4xCO2 (uptake shrinks while warming grows) from 1pctCO2 (both grow), so it would probably fail the 1pctCO2 test.

### New model: two layers, written in anomalies

Primes are differences from the control run, so the present-day climate and $Q_0$ do not change:

$$C \frac{dT'}{dt} = F'_{net} - \gamma_\theta(s)\,(T' - T'_d) + \Delta Q_{transport,\theta}(s)$$

$$C_d \frac{dT'_d}{dt} = \gamma_\theta(s)\,(T' - T'_d)$$

- $T'_d$: the deep-ocean temperature anomaly. It warms slowly and gives the model memory.
- $C_d$: the heat capacity of the deep layer (large). Fitted by gradient, starting from a two-layer fit to NorESM2 (Section 14).
- $\gamma_\theta(s) \ge 0$: the exchange between the layers, predicted by the network (softplus output). Strong exchange where winds drive upwelling represents the ocean thermostat, because the upwelled water comes from the deep layer, which has not warmed yet.
- $\Delta Q_{transport,\theta}(s)$: the change in horizontal ocean heat transport, predicted by the network. We remove its global mean, because transport moves heat; it does not create it.

The full model is the two-layer model of Held et al. (2010) in each cell, with a learned, state-dependent exchange and a learned transport change.

**What this solves:**

- **Memory:** slow warming over decades, and uptake that becomes smaller as the deep ocean warms.
- **A clear split:** uptake is the exchange with the deep layer; transport is the zero-mean rest. In the old plan the split was not identifiable from the data.
- **Energy:** heat is conserved between the two layers.
- **Interpretation:** we can map where the network increases the exchange and check whether this matches the upwelling regions.

**Zero in the control:** at the control state, $T' = T'_d = 0$, so the exchange is zero. We also force $\Delta Q_{transport}$ to be zero at the control state, for example $f(s) - f(s_0)$. The network therefore does not change the present-day climate, and $Q_0$ keeps its role as the bias correction.

**Network inputs $s$** for each column: $T'$, the memory signal $T' - T'_d$, the surface wind stress (upwelling depends on the winds), the global-mean warming, latitude and longitude.

**Network:** a small MLP applied to each column (about 6 inputs, 2 hidden layers of 64). Option: a small CNN for neighbor information.

---

## 6. Data

| Data | Variables | Role | Status |
|---|---|---|---|
| NorESM2-LM, CMIP6 (Pangeo or ESGF) | `tos`, `hfds`, `tauu`, `tauv`; piControl, abrupt-4xCO2, 1pctCO2 (and abrupt-2xCO2 if it exists) | Training target: how ocean heat transport and uptake respond to warming | New: get before Dec 4 |
| NorESM2-LM, CMIP6 (optional) | `thetao`, averaged over roughly the upper few hundred meters | Check the deep-layer temperature | Optional |
| ERA5 | Surface fluxes, wind stress | Evaluate the present-day control; set the input scales | Have it (`week4/get_obs.py`) |
| NOAA OISST | Observed SST, 1982 to now | Evaluate the present-day east Pacific bias; stretch: observed trend | Public: get before Dec 4 |
| Phase I JEM runs | Q-flux control, 2x and 4xCO2, `tdland` 1 and 10 days | Baseline (fixed Q-flux) | Done (`results/w5-*`) |

Observed data (ERA5, OISST) gives only the present-day climate. It cannot teach how the ocean responds to warming. Only NorESM2 can do that.

**Check before Dec 4:** that NorESM2-LM has `hfds`, `tauu` and `tauv` for every experiment we need, and whether it ran abrupt-2xCO2. If it did not, we use the time when CO2 doubles in 1pctCO2 (about year 70) as the "different size of forcing" test.

### Training target from NorESM2

For a slab with the same heat capacity $C$, NorESM2 gives an implied ocean flux:

$$Q_{implied} = C \frac{dT_{NorESM}}{dt} - \text{hfds}$$

Calculate it for piControl and abrupt-4xCO2. Then

$$\Delta Q_{target} = Q_{implied}(\text{4xCO2}) - Q_{implied}(\text{piControl}).$$

Regrid to T31 (48 × 96). Use the piControl variability to find the noise in the target.

### Deep-layer temperature for offline training

NorESM2 has no "deep layer" in our sense. We calculate $T'_d$ by integrating the deep-layer equation (Section 5) driven by NorESM2's own SST anomalies, with $\gamma$ and $C_d$ from the two-layer fit. This needs no extra data. Optionally, we check it against NorESM2 `thetao`.

---

## 7. Experiments

| Run | Ocean | Purpose |
|---|---|---|
| E0 | Week 5 fixed Q-flux | Baseline (done) |
| E1 | Two-layer, one global $\gamma$ and $C_d$, fitted by gradient | Physical baseline: the network must beat this |
| E2 | Two-layer, a map $\gamma(x)$, fitted by gradient | Spatial physical baseline |
| E3 | Network, trained offline | ML, offline |
| E4 | Network, partially online | ML, middle option (Section 8) |
| E5 | Network, fully online | ML, full gradient (Section 8) |
| Ablations | Remove the memory input; remove the wind stress | Which information the ocean needs |

E1 and E2 are important. Without them, a reviewer can say that a simple two-layer ocean gives the same improvement as the network.

The ablations answer "what does the learned ocean need: memory, winds, or both?" Every outcome is a result.

**Train and test split:** train on abrupt-4xCO2. Test on 1pctCO2 (a different rate of forcing) and on 2xCO2 (a different size of forcing; Section 6).

**Base setup:** use `tdland` = 10 days, so that land can warm (Section 4).

---

## 8. Training modes and loss

### The main rule: compare climate, not weather

JEM and NorESM2 have different weather, so their months do not match. Comparing JEM's SST in one month with NorESM2's SST in the same month mostly compares noise. Every loss therefore uses **climate statistics**:

- the $\Delta$SST pattern per K of global warming, averaged over time windows after the CO2 increase (for example years 1–5, 10–20 and 30–50 of abrupt-4xCO2), so that the timing of the warming is included, not only its final pattern;
- or ocean heat tendencies compared with $\Delta Q_{target}$ for the same state, averaged over many windows.

### Physical check in the loss

We add a small penalty that keeps the learned flux close to $\Delta Q_{target}$:

$$\mathcal{L} = \mathcal{L}_{SST\ statistics} + \lambda\,\lVert \Delta Q_\theta - \Delta Q_{target} \rVert^2$$

We report two scores separately: SST skill, and how close the learned flux is to the NorESM2 flux. This catches **compensating errors**: the JEM atmosphere has its own errors, and online training can "fix" SST by learning a wrong flux. If SST improves but the flux becomes unphysical, that is also a result.

### The three modes

| Mode | How it works | Strength | Risk |
|---|---|---|---|
| **Offline (E3)** | Fit the network to (NorESM2 state, $\Delta Q_{target}$) pairs, then put it in JEM | Fast and simple (minutes) | The JEM atmosphere differs from NorESM2's, so the inputs move out of the training range. The run can drift or blow up (the known offline problem) |
| **Partially online (E4)** | Run JEM forward and save its daily surface fluxes and winds. Train the ocean part by replaying only the ocean equations over years with these saved fluxes; the loss uses multi-year SST statistics. Then run JEM again with the new network to update the fluxes, and repeat | Long windows are cheap (replaying the ocean is fast) and gradients are stable, because they do not go through the chaotic atmosphere. Atmosphere feedback enters through the outer loop | The feedback is updated only between rounds |
| **Fully online (E5)** | Backpropagate through the whole JEM, including the atmosphere, in short windows (a few days, inside the ~4-day reliable range from Week 6), with `jax.checkpoint`. The loss compares ocean heat tendencies with the target, averaged over many windows | The most complete gradient | Chaos in the atmosphere, compute cost |

The partially online mode is similar to iterative retraining under distribution shift (as in DAgger, Ross et al. 2011): the training data is made again with the model's own behavior. It is the practical main mode. Fully online is the most ambitious mode.

**We compare the three modes on:** stability (drift, blow-ups), warming-pattern skill on the 1pctCO2 test, how physical the learned flux is, and compute cost.

---

## 9. How we measure success

**Main metrics**

| Metric | NorESM2 target | JEM now (Week 5) |
|---|---|---|
| East Pacific warming per K, east / west | 1.09 / 0.90 | 1.57 / 0.97 |
| Subpolar North Atlantic warming per K | 0.40 | about 1.0 |
| Southern Ocean warming per K | 0.29 | about 1.0 |
| Time to 90% of the warming in abrupt CO2 | Decades | 4 years |
| Pattern correlation of $\Delta$SST / $\Delta T$ with NorESM2 | 1 | Calculate in week 1 |

The time-to-90% metric is only reachable with the deep layer (Section 5).

**Secondary metrics**

- How close the learned flux is to $\Delta Q_{target}$ (physical realism; Section 8).
- Stability of each training mode (drift, blow-ups).
- Western North America `tas` and `pr` (with the land caveat).
- The Gregory slope (the pattern effect on climate sensitivity).
- The present-day cold-tongue bias against OISST.

**Not used as targets:** Arctic warming and the land/ocean ratio. They mainly reflect sea ice and `tdland`, not the ocean.

**Noise:** JEM has its own natural variability. An improvement counts only if it is larger than this noise. We use long enough runs or several runs, and the piControl noise of the target.

**Hypothesis (draft):** the best online-trained network (E4 or E5) reduces the east Pacific east–west warming error by at least 50% compared with E0, and it beats E1 and E2 on the 1pctCO2 test. Change the numbers after the preliminary analysis.

---

## 10. Compute cost

**Running the model with the network: under 1% slower (estimate)**

- JEM updates the slab ocean once per day (daily coupling). The atmosphere takes a step every 30 minutes, so it runs 48 steps for each ocean update.
- The MLP on 3,051 ocean cells needs about 30 million operations per call. The atmosphere does about 1,000 times more work in one day.
- The deep layer adds a few operations per cell per day, so its cost is very small.
- On a GPU, JAX compiles the network into the same program as the model.
- Measure it in week 1: run 1 year with a zero network and 1 year without it.

| Hardware | Measured speed |
|---|---|
| NRP GPU (GTX 1080 Ti), JCM atmosphere only (Week 4) | 55.8 model years per hour |
| Laptop CPU (M2), JEM with slab (Week 5) | about 8 model years per hour |

**Training the network (estimates)**

| Mode | What it needs | Time (estimate) |
|---|---|---|
| Offline (E3) | Fit to a table of NorESM2 numbers; the climate model does not run | Minutes |
| Partially online (E4) | One JEM run per round to save fluxes, plus fast ocean-only replays | Each replay: seconds to minutes. Each round: one JEM run (about 4 hours for 30 years on the laptop; faster on GPU). About 5–10 rounds |
| Fully online (E5) | Forward run plus backward gradient for each training step | Hours |

For fully online training on the NRP GPU (estimates): each step costs about 3–4 forward runs (the backward pass costs 2–3 times the forward run, and `jax.checkpoint` runs parts of the model again). Windows of a few days are much cheaper than the 1-month windows in the old plan, but more windows are needed to average out the weather.

**Limits:** the GPU has 11 GB of memory, and the JupyterHub server has 8 GB RAM and 1 CPU core. Use NRP Kubernetes Jobs for long runs.

---

## 11. Team split

| Member | Area | Main output |
|---|---|---|
| A | Data: NorESM2 download, regrid to T31, $\Delta Q_{target}$, deep-layer $T'_d$ from NorESM2, two-layer fit, OISST and ERA5 checks | Training dataset and evaluation targets |
| B | Deep layer and baselines: add the deep layer to `run_slab.py`, fit E1 and E2, offline network (E3) | Physical baselines and offline results |
| C | Online training: replay harness and outer loop (E4), full gradients through JEM (E5), gradient checks | The three-way training comparison |
| D | Evaluation and communication: metrics, maps, ablations, western US analysis, speed-vs-accuracy figure, website | Results section, figures and website |

Annaly built the Q-flux control (Week 5 Task 2), so Annaly is a natural choice for B or C. Assign the others by strength.

---

## 12. Schedule (7 weeks)

The course asks for about 11 hours per person per week on the project.

| Week | A (data) | B (deep layer, offline) | C (online) | D (evaluation) |
|---|---|---|---|---|
| 1 | Regrid NorESM2 `tos`, `hfds`, wind stress to T31 | Add the deep layer and a zero network to `run_slab.py`; check that E0 is unchanged; time it | Gradient of SST statistics with respect to $\gamma$ and $Q$; compare with finite differences | Metric code; score E0 against NorESM2 |
| 2 | $\Delta Q_{target}$ for 4xCO2 and its noise from piControl; $T'_d$ from NorESM2 | Fit E1 (global $\gamma$, $C_d$) and E2 ($\gamma(x)$); run 30-year 4xCO2 | Build the ocean-only replay harness with saved fluxes | OISST and ERA5 present-day evaluation of E0 |
| 3 | 1pctCO2 (and 2xCO2) test targets | Train the offline network (E3) | First partially online rounds (E4) | Score E1 and E2 |
| 4 | Input scaling, extra input variables | Run E3 for 4xCO2 and 1pctCO2; check drift | E4 at full scale; first fully online tests in short windows (E5) | Score E3; east Pacific maps |
| 5 | Stretch data (OISST trend or Veros) | Ablations: remove the memory input; remove wind stress | E5 at scale, or fall back to E4 | Score E4 and E5; three-way comparison; speed-vs-accuracy figure |
| 6 | Report: data section | Report: model and baselines section | Report: training section | Western US analysis; website |
| 7 | Buffer and polish | Buffer and polish | Buffer and polish | Buffer and polish |

---

## 13. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Fully online gradients (E5) are unstable, even in short windows | Partially online (E4) is the main result; offline (E3) is also a complete result |
| Offline network blows up in JEM | That is a result: it shows why online training is needed |
| The network does not beat E1 or E2 | That is a result: a simple two-layer ocean is enough, and the missing physics was memory, not a complex response |
| Online training improves SST but learns an unphysical flux | That is a result: it shows compensating errors; the physics penalty and the two separate scores make it visible |
| The two-layer fit does not match NorESM2's slow warming well | Try two deep time scales, or a $\gamma$ that depends on the region; report the limit |
| NorESM2 variables are not on Pangeo | Use ESGF |
| NorESM2-LM has no abrupt-2xCO2 | Use the CO2-doubling time in 1pctCO2 |
| Western US land results are poor | Report them with the `tdland` caveat; the ocean results are the main result |
| The novelty claim is wrong | Ask Duncan now and change the framing |

Each fallback still gives a result.

---

## 14. Preliminary analysis before December 4

The course needs proof that we can get the new data, that it contains the needed information, that it has enough quality and quantity, and that we have the compute.

1. Discuss this plan with Duncan.
2. Download NorESM2-LM `tos`, `hfds`, `tauu` and `tauv` for piControl, abrupt-4xCO2 and 1pctCO2. Check whether abrupt-2xCO2 exists.
3. Calculate $\Delta Q_{target}$ in the east Pacific. Show that it is larger than the piControl noise.
4. **Fit the two-layer model to NorESM2's global-mean abrupt-4xCO2 response** (the method of Geoffroy et al. 2013). This takes minutes. It gives starting values for $\gamma$ and $C_d$, and it shows that the slow time scale is reachable.
5. Download OISST. Calculate the east Pacific climatology.
6. Take `jax.grad` of a multi-year east Pacific SST statistic with respect to $\gamma$ in an ocean-only replay, and of a short-window statistic with respect to $Q$ in full JEM. Compare both with finite differences.
7. Record the time and memory of step 6 on the NRP GPU.

Steps 3, 4 and 6 are the most important. Step 3 shows that the target exists. Step 4 shows that the deep layer can give the right time scale. Step 6 shows that both online modes are possible.

---

## 15. Proposal parts

Use the course LaTeX template on Overleaf.

- **Context (general audience):** simple climate models keep the ocean frozen in its present state, with no deep ocean. We teach the ocean to change as the climate warms, and we test whether this fixes the projected warming in the eastern Pacific, which matters for the West Coast.
- **Technical problem statement (expert):** the equations in Section 5, the E0–E5 matrix in Section 7, the training modes and loss in Section 8, the metrics in Section 9, and the related work in Section 17.
- **Primary output:** a report, plus a website with interactive warming maps for E0–E5.
- **Feasibility:** the results of Section 14.
- **Schedule:** Section 12, as a separate PDF.

---

## 16. Elevator pitch and communication

### Elevator pitch bullet points

- We need to know how the climate will change in each region, to plan for future changes and to help people connect global warming to real effects in their lives.
- California's water resources, fire risk and coastal fog depend on the eastern Pacific Ocean.
- Winds push the warm surface water away (west along the equator, offshore along the California coast), and colder deep water comes up to replace it. That cold water cools the West Coast and affects rain and El Niño.
- Big climate models simulate ocean currents well, but take weeks for a single run. Our mentor's lab works on small, faster models that can test many future scenarios. The fast model that we use treats the ocean as a thin, flat slab of water: the currents are fixed, and there is no deep ocean to store heat.
- Because of this, the eastern Pacific warms about 40% more relative to global warming than in a full model, and the whole model warms in years instead of decades.
- Can we keep a climate model fast but teach its ocean to respond to warming?
- We add a slow deep-ocean layer and a small neural network that learns how heat moves into the deep ocean and around the globe. Because our model is differentiable, we can calculate how each change affects each output, so we can train the network inside the running model.
- If this works, we expect to cut the error in the eastern Pacific warming pattern by about half, so a fast model can give more accurate regional answers, closer to the full, slower climate model.

The 40% comes from Week 5: JEM warms the east Pacific at 1.57 K per K of global warming, and NorESM2 at 1.09 (1.57 / 1.09 ≈ 1.44).

### Headline number (fill in after the results)

"Our fast model overestimated eastern Pacific warming by 44%. Teaching its ocean to respond to warming cut that to X%, while it runs about N times faster than the full model."

### Key figures

- **Speed vs. accuracy:** model-years per hour against warming-pattern error, with points for NorESM2, frozen JEM, the physical baselines and the learned JEM. One figure that anyone understands.
- **Consequence chain:** eastern Pacific warming pattern → low clouds and the pattern effect (how much the whole planet warms) → El Niño and California rain → water supply and fire risk.
- **Three-way training comparison:** offline vs. partially online vs. fully online, on stability and skill.

### Everyday analogies

- The frozen ocean is like a thermostat stuck on one setting.
- The deep layer is a slow battery that stores heat for decades.
- The eastern Pacific is the planet's air conditioner, and the frozen model turns it off too early.

### Website

- Side-by-side maps for the frozen ocean, the learned ocean and NorESM2, with a slider over time or CO2 level.
- A short animation of the warming pattern improving as the network trains.
- "Runs on a laptop": because the model is fast, anyone can explore many scenarios.

---

## 17. Related work to read and cite

Check each one before the proposal. The proposal must state how this project differs from them.

- **Kochkov et al. (2024), NeuralGCM:** learned physics trained online through unrolled model steps.
- **The Ai2 climate modeling group:** ML corrections in FV3GFS, and ACE2-SOM (an ML atmosphere coupled to a slab ocean with a prescribed Q-flux and mixed-layer depth; arXiv 2412.04418).
- **"A Framework for Hybrid Physics-AI Coupled Ocean Models" (arXiv 2510.22676):** why offline ML parts fail when coupled, and the online-training alternative.
- **Frezat et al. (2022):** offline ("a priori") and online ("a posteriori") training of closures.
- **Held et al. (2010)** and **Geoffroy et al. (2013):** the two-layer model of fast and slow warming, and how to fit it.
- **Ross et al. (2011), DAgger:** iterative retraining under distribution shift (background for the partially online mode).
- **Clement et al. (1996):** the ocean dynamical thermostat.
- **Seager et al. (2019):** the observed east Pacific trend that models do not reproduce.
- Literature on the pattern effect and on slab-ocean Q-flux experiments.

---

## 18. Questions for Duncan

1. Is a learned, state-dependent ocean (deep-layer exchange plus transport change) trained online in JEM new, or do you know of similar work?
2. Does JEM already have a deep-layer option, or is adding one to the slab code straightforward?
3. Is the NorESM2 implied Q (from `tos` and `hfds`) a good training target, or is there a better one? Is `thetao` worth using to check the deep layer?
4. Does the partially online approach (replaying the ocean with saved atmosphere fluxes, then updating them) sound reasonable?
5. Is `tdland` = 10 days the best land setup, or is there a more stable fix?
6. Is the comparison with JEM's Veros coupling realistic as a stretch goal?
7. What compute should we use for long runs: NRP Kubernetes Jobs, or Casper?
