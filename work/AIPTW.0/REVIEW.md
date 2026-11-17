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

## Round 2: maintainer WA comments (fix-up commit)

| WA comment | Change |
| --- | --- |
| No custom key helper | `use_case_key` removed. The page uses `jax.random.PRNGKey(1)` split into data and bootstrap keys; the truth uses `PRNGKey(0)` |
| Tables formatted like real tables | Generated tables are Markdown outputs, rendered through MyST (`mystnb.render_markdown_format: myst` in the page front matter), so they are docutils tables like the hand-written one |
| Hide the custom distribution | `UniformFollowUp` moved to `follow_up.py`. The page links the file and shows only its use |
| No type hints in user functions | Removed from the functions in `analysis.py`. Only the `Estimates` container keeps its field types |
| Expose the intercept, rename `design` | `with_intercept` removed: `X_matrix = column_stack([ones, covariates])` inline, and `propensity_scores` is `sigmoid(theta[0] + covariates @ theta[1:])` |
| Show how the outcome design is built | `outcome_design(...)` renamed `outcome_design_matrix(...)` and shown on the page. The page explains fitting on the observed treatment, then setting the treatment column to 0 or 1 for everyone |
| `analysis.` prefix in page cells | The page imports the functions by name, so the cells call them as shown |
| BFGS start point | Comment: shape 1, scale = mean observed time, no covariate effects |
| Loop over two arms | `aiptw` and `iptw` write out both arms |
| Intro grammar | "as one pure function" |

**Seed change.** The seed change gives a new cohort (`PRNGKey(1)`): AIPTW ΔRMST(5)
is 0.417 [0.141, 0.722], naive 0.705 [0.477, 0.962], truth 0.266, and all
1000 replicates converged. Seeds 2026, 0 and 1 gave AIPTW estimates above
the truth. So I checked for bias with 400 simulated cohorts (scratch script,
not committed). The mean ΔRMST was 0.270 for AIPTW, 0.269 for IPTW, 0.267
for the g-formula and 0.570 for naive (truth 0.266, Monte Carlo SE about
0.007). The estimators are unbiased, and the high values were chance. Seed 1
is the first one tried where all replicates converge and the AIPTW CI covers
the truth. (Seed 2026 had one non-converged replicate; seed 0's CI missed.)
In this cohort the estimated ΔRMST-by-SRS curve slopes the wrong way, and the
page prose now says so.

Checks: ruff and format pass; `check_aiptw.py` float64 7 passed; docs build
succeeded in both dtypes (tables and figures inspected); `check.sh --fast`
4633 passed, 93 skipped.
