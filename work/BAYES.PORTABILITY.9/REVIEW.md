# BAYES.PORTABILITY.9 — Standalone notebook handoff

Independent review: round 1/3

## Read first

1. PLAN.md and DEVIATIONS.md: maintainer approved the notebook-local draw
   calculation in this conversation; deviation status remains open for them.
2. Notebook setup/trial cells and its own running instructions.
3. tests/unit/check_bayesian_run.py, then the unchanged model.py and shared
   examples/weibull_simulation/study.py generator.

## Invariants and changes

The notebook imports installed dependencies only. It no longer imports sys,
Path or examples modules, discovers a repository root, or mutates sys.path.
Its short local draw retains the folded data key, U-then-C split, float64
promotion, Weibull inverse transform, PH scale, censoring limit, arm order,
event comparison and validation. It uses the recorded calibrated limit rather
than repeating the host calibration algorithm. The runner remains shared.
The notebook's own instructions and documentation links work outside a checkout.
No library API, dependencies, spec or protected files changed.

## Checks

- `scripts/check.sh --fast`: exit 0, all gates passed with the committed feature
  selection. Ruff: 305 files formatted, all checks passed; mypy: 25 source
  files clean. Implementation x64: 1219 passed/8 skipped; x32: 1221 passed/6
  skipped. Doctests: 4 passed per dtype. Spec x64: 4654 passed/46 skipped;
  x32: 4597 passed/93 skipped, two existing NumPy log-zero warnings per mode.
  Tier 2 and docs build were not run for this notebook code step; the full
  notebook was executed by its optional check instead. Final test-only string
  formatting was checked again after the gate's formatting phase.
- `uv run --group docs --group examples --group test pytest -q
  tests/unit/check_bayesian_run.py`: 3 passed, 1 warning in 59.55s. This includes
  exact time/event/arm/limit equality with make_trial (spec 11.3 reproduction),
  fresh-kernel full notebook execution from tmp_path, and the unchanged
  bit-for-bit committed-posterior reproduction check.
- The outside-checkout test prepends an assertion that the examples package
  cannot be found, before any notebook cell executes. Source outputs are not
  overwritten. IPython reported ~/.ipython unwritable and used a temporary
  directory; execution passed.
- Initial exact-trial check failed on 53 times by at most 1.78e-15 because
  Python scalar versus strong array operations differed. Matching the shared
  generator's explicit dtype promotion/broadcasting fixed exact equality;
  no tolerance was introduced. Initial run: 1 failed, 2 passed in 25.41s.
- Ruff format/check and git diff --check pass after final edits. A format
  check caught a split string after the test assertion was added; formatting
  corrected it.
- The two new checks are optional `check_` tests, explicitly invoked with
  docs/examples dependencies. They are not collected by the normal gate or
  slow-test measurement script, so tests/slow_tests.txt is unchanged.

## Remaining sub-step / limits

This is the code step. A separate docs step still needs to update the earlier
checkout-only claims in docs/use-cases/bayesian-rmst.md and the example README;
the notebook's own instructions already describe standalone use. See PLAN.md
Docs to-do. Stop after this step's review per the working agreement.
The approved deviation remains open for maintainer resolution. No dependency
installation in a brand-new environment was tested; execution used the locked
project environment, with the notebook working directory outside the checkout.
The pre-existing untracked nvim.log (from the previous diff hook) was preserved
at /tmp/survjax-bayesian-use-case-nvim.log to restore a clean review precondition.
No spec edits, publishing or integration performed.
