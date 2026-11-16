# AIPTW.0 review notes

## Read first, in order

1. `examples/aiptw_rmst/cohort.py`: the simulation and `UniformFollowUp`.
2. `examples/aiptw_rmst/analysis.py`: models, estimators, `analyse` and `bootstrap`.
3. `docs/advanced/aiptw-rmst.md`: the tutorial. It executes everything live.
4. `tests/unit/check_aiptw.py`: optional checks.

## Invariants the code relies on

- **Censoring weights.** `G(min(t, τ)) ≥ G(5) = 0.75 > 0`, so the weights are
  finite with no substitution. Patients censored before τ get weight 0;
  everyone else gets `1/G`.
- **`UniformFollowUp` safe inputs.** It substitutes the support midpoint
  before every log (spec 7.2), so the branches outside the support produce
  neither NaN values nor NaN gradients.
- **Values beyond `end`.** Following the spec 4.5 row "S(t)=0 beyond a finite
  support end", the family's `log_hazard` override returns its left limit,
  +inf, and `log_prob` is −inf.
- **Purity of `analyse`.** It is a pure function of the `Cohort` pytree, with
  no keys and no globals except the constants `TAU` and `FOLLOW_UP`. That
  purity is what makes `vmap` over resampled indices valid.
- **Convergence.** Non-convergence is reported in `Estimates.converged`
  (optimistix `throw=False`). On the page, all 1000 replicates converge.

## Checks and what they show

| Check | Shows |
| --- | --- |
| `test_uniform_follow_up_matches_closed_forms` | S, f and h of Uniform(4, 8) on both sides of each edge, by hand |
| `test_uniform_follow_up_gradients_are_finite_on_both_sides_of_the_support` | the gradient in `end` equals the hand derivative, with no NaN from outside points |
| `test_propensity_likelihood_is_the_bernoulli_formula` | NumPyro's Bernoulli sum equals Σ[Aη − log(1+e^η)] |
| `test_fits_reach_zero_score` | both fits converge, and the outcome score is below 1e-5 |
| `test_aiptw_with_known_propensity_and_no_censoring_is_the_mean_per_arm` | the estimator algebra, by hand |
| `test_censoring_weights_drop_unknown_status_and_upweight_the_rest` | the IPCW weights, by hand |
| `test_vmapped_bootstrap_matches_a_loop` | `jit(vmap(analyse))` equals a loop of `analyse` (rtol 1e-8) |

## Checks run

- `ruff format --check .` and `ruff check .` both pass.
- `mypy src` reports no issues.
- `tests/unit/check_aiptw.py`: float64 7 passed; float32 7 skipped (the
  tutorial runs in float64).
- `scripts/build_docs.sh` succeeded in both dtypes. I read the rendered HTML
  outputs and looked at the four figures.
- `scripts/check.sh --fast`: 4633 passed, 93 skipped, all gates passed.
- **Not run:** the full non-fast `scripts/check.sh`. Also not run: the
  `independent-review` skill, which is not available in this session.

## Unsure about

- **The page sets `jax_enable_x64` itself,** in a hidden cell, so in the x32
  docs build it still runs in float64. This matches its float64-only checks.
- **The SRS shows a chance imbalance of +0.23 SMD** in this seeded cohort. The
  page states it plainly rather than re-seeding.
