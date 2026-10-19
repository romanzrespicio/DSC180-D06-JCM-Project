# Week 3: answers to the form questions

Scores come from `results/leaderboard.csv` (written by `notebooks/Week3_Leaderboard.ipynb`).
The figures and supporting tables are in `notebooks/Week3_Presentation.ipynb`.

## Q1. Our leaderboard next to the published one, and whether we reproduced it

Total NRMSE over 2080–2100 (published values in brackets):

| Model | tas | dtr | pr | pr90 |
|---|---|---|---|---|
| Pattern scaling | 0.320 (0.320) | 18.55 (19.72) | 3.34 (3.66) | 4.13 (4.46) |
| Pattern scaling, EBM-predicted T | 0.383 (–) | 18.41 (–) | 3.45 (–) | 4.23 (–) |
| Gaussian process | 0.299 (0.478) | 20.52 (22.58) | 3.91 (4.05) | 4.65 (4.70) |
| Random forest | 1.475 (0.400) | 16.69 (22.46) | 5.50 (5.03) | 6.09 (5.40) |
| CNN-LSTM | 0.363 (0.327) | 13.66 (16.78) | 3.36 (3.17) | 4.16 (4.34) |
| Variability floor (ours, approx.) | 0.235 | 7.13 | 2.14 | 3.01 |

- **Pattern scaling:** tas reproduced exactly. dtr, pr, and pr90 are 6–9% better because we train on
  6 scenarios, including hist-GHG and hist-aer, and the official setup uses 4. A replica of the official
  setup matches all four published scores to 3 decimals. The published version gets the true ssp245
  global temperature as input. Our EBM version predicts that temperature from the forcings, and it
  costs little (tas 0.32 → 0.38).
- **Gaussian process:** tas is better (0.30 vs. 0.48) because of a linear CO₂ kernel that we added.
  Without that kernel, tas global matches the paper (0.0736 vs. 0.0738). dtr is slightly better, and
  pr and pr90 are reproduced.
- **Random forest:** tas is not reproduced (3.7× worse). Leave-one-scenario-out CV chose
  `max_features='sqrt'`, which causes most of the spatial error. Training on 6 scenarios instead of 4
  causes most of the global error: the forest's warming levels off at about 1.4 K and is about 0.5 K
  too cold over 2080–2100. Even with the paper's exact setup, we get tas global 0.100 vs. 0.058, and
  we did not find the rest of the cause. dtr is better, probably because hist-aer is in the train
  data. pr and pr90 are 9–13% worse.
- **CNN-LSTM:** reproduced. The differences are within the spread over 5 seeds. dtr is better
  (13.7 vs. 16.8).

## Q2. Which target is hardest, and why

DTR is the hardest. Its global-mean change in ssp245 is about zero (−0.01 K), because greenhouse gases
shrink DTR (nights warm more than days), and the decline in aerosols widens it (more sunlight reaches
the surface). The two effects cancel globally but not regionally, so the pattern depends on where the
aerosols change. DTR has the lowest global signal-to-noise ratio (3.3, vs. 31 for tas), and the NRMSE
divides by a near-zero global mean. Even our best DTR emulator stays 1.9× above the variability floor.

Precipitation (pr, pr90) has the lowest local signal-to-noise ratio (about 6 per grid cell vs. 34 for
tas): rainfall varies strongly from year to year, and aerosols shift it regionally. tas is the
easiest: a large, smooth signal far above the noise.

## Q3. Ensemble spread relative to our errors

We scored each ssp245 member against the mean of the other two, and we scaled the result to estimate
the score of a perfect emulator against the 3-member mean. That floor is about 0.23 (tas), 7.1 (dtr),
2.1 (pr), and 3.0 (pr90).

Our best models (pattern scaling, GP, CNN-LSTM) are 1.3–1.9× the floor, so much of the remaining error
is internal variability that no emulator can remove. tas has little room left for improvement, and dtr
has the most. The random forest is the outlier on tas (6.3× the floor), because of a systematic cold
bias, not noise.

This floor is lower than the paper's "Variability" row (tas 0.23 vs. 0.41), because that row measures
the member spread, not the best score that an emulator can get.

## Q4. What the AI assistant got wrong, or nearly wrong, and how we caught it

- pip upgraded numpy past the pin, and TensorFlow then crashed with conda's pyarrow. We caught it with
  import and smoke tests.
- The GP targets were float32, but GPflow needs float64. We caught it with a short test fit before the
  full run.
- The assistant read the official GP code before we asked it to work without it. It told us, and we
  used the official code only for the later comparison.
- Pattern scaling: the assistant proposed to drop the intercept because an ablation scored better on
  ssp245. We saw that this was tuning on the test set, and we kept the original choice.
- JAX: the notes said that float32 differs "at about 1e-6". We measured it (2e-7) and corrected the
  text.
- Random forest: the first search space allowed `min_samples_leaf` of 1 and 2, which needs about 20 GB
  for 200 trees on a 16 GB laptop. We caught it by timing and measuring one small forest before the
  full run.
- Random forest: the assistant planned to score the CV folds with the shared NRMSE, which needs
  2080–2100, but the historical folds end in 2014. We caught it by reading `metrics.py` before we wrote
  the CV.
- Random forest: after the poor tas score, the assistant proposed a second "paper replica" RF for the
  leaderboard. We rejected it, because the proposal came from the ssp245 score.
- Random forest: the assistant described the RF's global-mean error from a diagnostic run ("jumps
  1 K"), not from the submitted model, which levels off and is too cold. We caught it when we plotted
  the submitted model.
- CNN-LSTM: *(owner to add)*
