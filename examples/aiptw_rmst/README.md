# Doubly robust causal survival analysis

Does adjuvant chemotherapy for stage III colon cancer buy patients time? This
example estimates the effect on 5-year restricted mean survival time from a
simulated observational cohort with AIPTW. The analysis runs in full,
including a 1,000-replicate bootstrap. Read it on the
[docs page](../../docs/advanced/aiptw-rmst.md), or run it in the
[Jupyter notebook](aiptw-rmst.ipynb), which defines every analysis function in
its own cell. All data are synthetic, and the stromal radiomic score is a
fictional biomarker.

- `cohort.py`: the simulated cohort and the true values.
- `follow_up.py`: the administrative follow-up as a custom survjax
  distribution (`UniformFollowUp`).
- `analysis.py`: the propensity and outcome models, the estimators, the
  one-argument `analyse(data)` and the vmapped `bootstrap`.
- `reporting.py`: the table helpers of the page and the notebook.

From the repository root, open the notebook, or run the checks:

```sh
uv run --group docs --group examples jupyter lab examples/aiptw_rmst/aiptw-rmst.ipynb
JAX_ENABLE_X64=1 uv run --group examples pytest -q tests/unit/check_aiptw.py
uv run --group docs --group examples pytest -q tests/unit/check_aiptw_notebook.py
```

survjax itself does not depend on the `examples` group.
