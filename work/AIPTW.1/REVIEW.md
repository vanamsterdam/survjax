# AIPTW.1 review notes

## Read first, in order

1. `examples/aiptw_rmst/analysis.R`: the R port. Compare it side by side
   with `analysis.py`.
2. `examples/aiptw_rmst/benchmark/README.md`: the results and how to read them.
3. `examples/aiptw_rmst/benchmark/benchmark.py`, `benchmark.R` and `run.sh`.
4. `examples/aiptw_rmst/export_cohort.py` and `tests/fixtures/aiptw_rmst/`.

## Invariants

- **Same data.** R reads the CSV that survjax wrote at 17 significant digits,
  which round-trips float64 exactly. `test_r_fixture_cohort_is_the_tutorial_cohort`
  checks this with `assert_array_equal`.
- **Same model.**
  - survreg's Weibull AFT form is the Weibull PH model: shape `1/scale` and
    per-patient scale `exp(lp)`.
  - The RMST closed form is `b Γ(1+1/k) P(1/k, (τ/b)^k)`, obtained by
    substituting `x = (t/b)^k` into `∫₀^τ exp(-(t/b)^k) dt`.
  - `test_analysis_matches_r_survreg_on_the_same_cohort` checks all four
    estimators at rtol 1e-8; the observed difference is about 1e-9.
- **Fair timing.**
  - JAX compilation is timed separately, via `jit(...).lower().compile()`.
  - JAX uses its own `analyse_resamples(data, indices)` rather than
    `analysis.bootstrap`, because the latter re-jits a new closure on every
    call.
  - The bootstrap sizes (120, 1,200 and 12,000) divide over 12 devices.

## Checks run

- `ruff format --check .` and `ruff check .` pass. `scripts/check.sh --fast`: 4633 passed, 93 skipped, all gates passed.
- `tests/unit/check_aiptw.py` in float64: 9 passed (7 old, 2 new).
- `examples/aiptw_rmst/benchmark/run.sh` ran to completion (exit 0) on an
  otherwise idle machine. Its results are in the benchmark README.
- `scripts/fixtures/generate_aiptw_r_estimates.R` ran under R 4.4.2 and
  survival 3.7.0.

## Unsure about

- **Fixture loading.** `check_aiptw.py` reads the fixture with `np.loadtxt`
  and `json`, not through the `r_fixture` helper in `tests/conftest.py`. That
  helper is manifest-checked for the existing R sets. This optional examples
  check runs outside the gate, so I did not extend the helper or add a
  `SHA256SUMS`. The JSON records the cohort's MD5 instead.
- **The fixture direction is the reverse of the other fixtures:** survjax
  writes the data and R reads it.
- **The R 12-core intervals differ from the 1-core ones** because the forked
  processes use L'Ecuyer streams. Both are valid bootstrap draws.
- **Not measured:** a GPU, and a damped-Newton JAX solver. I tried plain
  optimistix Newton in a scratch script; it diverges from the tutorial's
  starting values.
- **The independent-review skill** is unavailable in this session, so it was
  not run.
