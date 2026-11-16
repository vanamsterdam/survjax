# AIPTW.0: AIPTW survival tutorial (plan and record)

- **Step ID:** AIPTW.0. One step at the maintainer's request ("just produce
  the demo"). The plan stays local; it does not move to `spec/use-cases/`.
- **Feature:** an advanced docs tutorial with example code. No library
  change, no new dependency, no `spec-features.txt` change.
- **Branch:** `aiptw`. **Step starting commit:** `69b796fa9192fb382bfd38bcbd80e3a67078fa0d`.
- **Spec sections used (not changed):** 4.1 and 4.5 (custom family and the
  direct-method value policy, including the "S(t)=0 beyond a finite support
  end" row), 6 (right-censored likelihood), 8.7 (staged PH-Weibull RMST), and
  3.2 (`SurvivalData` indexing for resampling).

## Maintainer decisions

1. The setting is stage III colon cancer, adjuvant chemotherapy versus surgery
   alone, with N = 500. The endpoint is the marginal ΔRMST at τ = 5 years.
2. There are three covariates. **Age** and **frailty** are confounders. The
   **stromal radiomic score (SRS)** is a fictional imaging biomarker. It is
   computed retrospectively on archived CT, so it is independent of
   treatment. It is mildly prognostic and mildly modifies the effect.
3. Censoring is purely administrative, uniform follow-up on [4, 8] years. It
   is implemented as a custom survjax family, `UniformFollowUp`, and it is
   known by design rather than fitted.
4. The propensity model is a logistic regression using NumPyro's
   `Bernoulli(logits=...)` likelihood. The outcome model is a Weibull PH fit
   with a survjax likelihood. Both are fitted with optimistix BFGS.
5. The analysis is a single `analyse(data)` function; the bootstrap is
   `jit(vmap(analyse))`.
6. Simulation values come from a short tuning run that is not part of the
   tutorial. The tutorial lives in a new docs section, "Advanced survival
   analysis tutorials".

## Simulation values (tuned, then fixed)

- Age is N(66, 10²), scaled as z = (age − 66)/10. Frailty is
  Bernoulli(sigmoid(−1.4 + 0.6 z)). SRS is N(0, 1).
- Treatment: `logit P(A=1) = 0.7 − 0.6 z − 1.0 frail`, with no SRS term.
- Death: Weibull PH with shape 1.3 and baseline scale 8.4 years. The log
  hazard ratios are log 0.70 for chemo, 0.35 for z, 0.70 for frail, 0.20 for
  SRS and 0.15 for chemo × SRS.
- Follow-up: Uniform(4, 8) years. Censoring weights up to τ are at most 4/3.
- **Tuning results** (large Monte Carlo, plus the seeded cohort):
  - about 61% treated, with estimated propensity between 0.12 and 0.95;
  - 5-year survival about 0.54 without chemo and 0.64 with it;
  - about 13% censored before 5 years;
  - true ΔRMST(5) is 0.266 years; the naive estimand is 0.591 years.
- **Tuning that was tried and rejected:** the first proposal used coefficients
  −1.0 (age) and −1.5 (frail). Its propensity tails were too extreme (min
  0.003 in a large sample), so the treatment model was softened.

## Files

- `examples/aiptw_rmst/cohort.py`: `Cohort` pytree, `resample`,
  `UniformFollowUp`, the simulation, and the Monte Carlo truth.
- `examples/aiptw_rmst/analysis.py`: the propensity and outcome fits,
  per-patient counterfactual RMST, censoring weights, the naive, g-formula,
  IPTW and AIPTW estimators, `analyse` and `bootstrap`.
- `examples/aiptw_rmst/README.md`: a short pointer.
- `tests/unit/check_aiptw.py`: optional checks in the `check_*` convention,
  run explicitly with the examples group.
- `docs/advanced/aiptw-rmst.md`: the MyST page. It runs the whole analysis
  live, including 1,000 bootstrap replicates (about 5 s).
- `docs/index.md`: the new toctree section and a link from the intro.

## Deviation from the earlier draft

- The smoke/full configurations, `run.py` and the committed `results/` are
  dropped. The full run is fast enough to execute live on the page.
- The separate `check_demo.py` is dropped in favour of `tests/unit/check_aiptw.py`.
  The examples are showcases, and their checks live in tests.

## Docs to-do

- None beyond this step. A runnable `.ipynb` copy, like the Bayesian
  example's, is a possible follow-up.
