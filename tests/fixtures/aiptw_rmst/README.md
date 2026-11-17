# AIPTW tutorial cohort, shared with R

The data direction is the reverse of the other fixtures. **survjax** simulates
the cohort, and R analyses it.

- `cohort.csv`: the 500 synthetic patients of the tutorial page, with columns
  `time, event, treated, z_age, frail, srs`, at 17 significant digits. It was
  written by `examples/aiptw_rmst/export_cohort.py` from
  `simulate_cohort(data_key, n=500)`, where `data_key` is the first half of
  `jax.random.split(jax.random.PRNGKey(1))`. That is the page's key, in float64.
- `r_estimates.json`: per-arm RMST(5) from the R port
  `examples/aiptw_rmst/analysis.R` (glm and survreg). It was generated with
  `scripts/fixtures/generate_aiptw_r_estimates.R` under R 4.4.2 and survival
  3.7-0, and it records the cohort's MD5.

`tests/unit/check_aiptw.py` checks that the CSV is exactly the simulated
cohort, and that the survjax analysis matches R to a relative 1e-8. The
observed difference is about 1e-9. To regenerate, from the repository root:

```sh
JAX_ENABLE_X64=1 uv run --group examples python -m examples.aiptw_rmst.export_cohort \
    tests/fixtures/aiptw_rmst/cohort.csv --n 500
Rscript --vanilla scripts/fixtures/generate_aiptw_r_estimates.R
```
