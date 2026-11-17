# DOCS.BAYES.9 — Clarify the Bayesian RMST tutorial (issue #9)

Step starting commit: 54e3fa9f280f63bcd9c08865fc99e128cd69c370
Worktree base: 54e3fa9f280f63bcd9c08865fc99e128cd69c370 (branch created from main).
Previous step/integration: same SHA, Merge PR #11 tooling improvements.
Branch: bayesian-use-case. Feature: documentation for the existing Bayesian demo.
Dependencies: the committed Bayesian example, posterior and reproduction check;
[issue #9](https://github.com/vanamsterdam/survjax/issues/9).
Completed feature selection: unchanged, scripts/spec-features.txt.

## Scope

One documentation topic, under 300 changed implementation/documentation lines:
edit docs/use-cases/bayesian-rmst.md, the explanatory Markdown cells of
examples/bayesian_rmst/bayesian-rmst.ipynb and its README.md. Explain the
per-observation factor shape, plate bookkeeping, overcounting from summing
inside the plate, and explicit data/covariate subsampling and scaling. Explain
draw/arm/time axes, float64 as committed-trial reproduction, and notebook
portability within a full checkout. Shorten the optional deterministic route
on the page by referring to the existing model instead of repeating its body.
Clarify Weibull keyword conventions locally; do not introduce a censored-wrapper
tail comparison without version-specific evidence. Keep the shared generator.

No runtime/example code, public API, dependency, spec or protected-path changes.
No new tests: use existing example checks and execute the unchanged notebook.
Spec coverage: 3.3, 5, 8.7, 11.3 and the Bayesian use-case main/details documents.
The issue explicitly requests these explanations; the model remains unchanged.

## Validation

- Verify NumPyro 0.22.0 plate/factor implementation and official documentation.
- Check a small fixed-parameter factor example demonstrates correct full-data
  reduction, scalar overcounting and subsample scaling.
- Execute the notebook in a fresh kernel from its own directory, without saving
  outputs into the source; run tests/unit/check_bayesian_run.py explicitly with
  optional examples dependencies (reproduces saved draws and acceptance checks).
- Run scripts/build_docs.sh directly (both dtypes), inspect rendered guidance,
  and run scripts/check.sh --fast before commit and independent review.

## Docs to-do

This step covers all issue #9 guidance. No follow-up code step is planned;
standalone notebook distribution and comparisons with other wrappers remain
outside this issue's required implementation scope.
