# BAYES.CENSOR20.9 — Use a round 20-year censoring limit

Step starting commit: b4817e7a916c847cee3c39806f3f2cba70f1c644
Worktree base: 54e3fa9f280f63bcd9c08865fc99e128cd69c370.
Branch: bayesian-use-case. Previous step:
[DOCS.BAYES.PORTABILITY.9](../DOCS.BAYES.PORTABILITY.9/PLAN.md).
Feature selection: unchanged. Dependency: existing trial/notebook reproduction
checks and standalone trial implementation.

The maintainer requests censor_limit = 20.0 before merging PR #12, after the
scope explained that this replaces the design's calibrated 55% limit and
requires regenerated JSON/NPZ/notebook results. That explicit assignment
authorizes the narrow exception recorded in DEVIATIONS.md; spec stays read-only.

Code step: change model.py's Bayesian trial limit to 20.0, remove its unused
calibration import, change the notebook trial cell/comment and horizon grid,
and refresh its own description/output. Regenerate results/posterior.npz and
metadata.json using run.py, preserving priors, seeds and sampler configuration.
Extend the existing optional exact-trial check with the requested 20.0 value
as an independent regression assertion; remove its unused tmp_path argument.
No library API, dependencies or protected paths change. Under 150 changed text
lines expected, excluding regenerated notebook image payloads and NPZ data.

Validation: runner acceptance, optional checks (exact equality, independent
fixed-limit assertion, outside-checkout execution and bit-for-bit posterior
reproduction), final notebook execution to refresh outputs, ruff and fast gate.
Write handoff, commit, independent-review, push to existing PR, then stop.
The merge request is withdrawn: do not merge into main.

## Docs to-do

Follow-up docs step: replace 19.2/164/55%/13-year text and calibrated-limit
language in docs/use-cases/bayesian-rmst.md and the example README; reconcile
recorded sampling claims with the regenerated metadata. Notebook-local prose
changes with this code step. See preceding step plans for context.
