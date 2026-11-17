# Doubly robust causal survival analysis

Does adjuvant chemotherapy for stage III colon cancer buy patients time? This
example estimates the effect on 5-year restricted mean survival time from a
simulated observational cohort with AIPTW. The page runs the whole analysis,
including a 1,000-replicate bootstrap; read it on the
[docs page](../../docs/advanced/aiptw-rmst.md). All data are synthetic, and
the stromal radiomic score is a fictional biomarker.

- `cohort.py`: the simulated cohort and the true values.
- `follow_up.py`: the administrative follow-up as a custom survjax
  distribution (`UniformFollowUp`).
- `analysis.py`: the propensity and outcome models, the estimators, the
  one-argument `analyse(data)` and the vmapped `bootstrap`.

From the repository root:

```sh
JAX_ENABLE_X64=1 uv run --group examples pytest -q tests/unit/check_aiptw.py
```

survjax itself does not depend on the `examples` group.
