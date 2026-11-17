# AIPTW.1: R port of the AIPTW tutorial and a mini benchmark (plan and record)

- **Step ID:** AIPTW.1, at the maintainer's request after AIPTW.0 round 3:
  "write an r implementation of the same analysis, using either survreg or
  flexsurv … let (surv)jax generate the datasets as a fixture … also run the
  small benchmark".
- **Feature:** examples, a benchmark and one fixture. No library change, no
  new Python dependency and no `spec-features.txt` change. R needs only
  `survival`, which is already pinned for the R fixtures.
- **Branch:** `aiptw`. **Preceding step:** AIPTW.0 (`work/AIPTW.0/`).
  **Step starting commit:** `c5a680efa63fbb00bb6fe1f3c10937f060e03fbc`.

## Choices

- **survreg, not flexsurv.** survreg's Weibull AFT form is the same model as
  Weibull PH. It is the fast, standard R tool, so the benchmark is fair to R.
  RMST uses the closed form `b Γ(1 + 1/k) P(1/k, (τ/b)^k)`.
- **Propensity model:** `glm(binomial)`, R's standard logistic regression.
- **Same structure as `analysis.py`:** one R function per Python function, with
  the same names, and `analyse(data)` and `bootstrap(data, n_boot, cores)`.
  The R file carries no tutorial explanation.
- **Data from survjax.** `export_cohort.py` writes CSV. The tutorial cohort
  (n = 500) is committed as a fixture. The larger benchmark cohorts are
  generated on the fly by `benchmark/run.sh`.
- **Benchmark grid:** cohort sizes 500, 5000 and 50 000, by bootstrap sizes
  up to 10 000. R runs on 1 and 12 cores (`mclapply`). JAX runs on 1 and 12 CPU
  devices, with the replicate axis sharded and compilation timed separately.

## Files

- `examples/aiptw_rmst/analysis.R`, `examples/aiptw_rmst/export_cohort.py`
- `examples/aiptw_rmst/benchmark/{benchmark.R, benchmark.py, run.sh, README.md}`
- `tests/fixtures/aiptw_rmst/{cohort.csv, r_estimates.json, README.md}`,
  `scripts/fixtures/generate_aiptw_r_estimates.R`
- `tests/unit/check_aiptw.py`: two more checks, for the fixture identity and
  the R agreement
- `examples/aiptw_rmst/README.md`: pointers

## Docs to-do

- The tutorial page is unchanged. Whether the benchmark belongs on the page is
  for the maintainer to decide.

## Withdrawn

The maintainer withdrew this step after round 4: "let's skip this entire R
comparison and just keep it to the survjax code with the default optimistix
bfgs solver; these benchmark investigations will have to come later".

- **The cleanup commit removes every file this step added** and restores
  `tests/unit/check_aiptw.py` and `examples/aiptw_rmst/README.md` to their
  versions at `c5a680e`.
- **For the later benchmark work,** the last commit with the benchmark is
  `83cbe8e`. It holds the R port, the benchmark scripts, the fixture and the
  results README; its "survreg's starting values" section has the Newton
  findings.
- **The cleaned-up saddle-free Newton solver** was shown to the maintainer
  in the conversation and is not committed anywhere.
