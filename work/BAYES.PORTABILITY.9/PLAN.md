# BAYES.PORTABILITY.9 — Make the Bayesian notebook portable

Step starting commit: 58f985f4b2e8ede0a0a669b6b225d39e690fdbc0
Worktree base: 54e3fa9f280f63bcd9c08865fc99e128cd69c370.
Branch: bayesian-use-case.
Previous step: [DOCS.BAYES.9](../DOCS.BAYES.9/PLAN.md).
Dependencies: existing shared generator and committed synthetic trial.
Completed feature selection: unchanged.

## Approved code step

Original issue #1 finding 7 asks for a copyable notebook without sys.path
manipulation; the preceding docs step only documented this limitation.
Remove sys/Path/root discovery and the make_trial import from the notebook.
Replace its trial-generation cell with explicit JAX inverse-Weibull and
Uniform censoring draws using the same folded key, random draw order and
recorded calibrated censoring limit. Leave the runner and shared generator
unchanged. Add an optional examples-environment test comparing the notebook
trial cell exactly against make_trial and executing the notebook from a fresh
temporary directory outside the checkout. No public library API or dependency
changes. Target under 100 changed implementation/test lines.

Spec coverage: 11.3 and use-case 02 main/details (shared generation and keys).
The maintainer approved the narrow shared-generator exception in this
conversation. See [DEVIATIONS.md](DEVIATIONS.md); its status remains open for
the maintainer to resolve. Notebook prose and links change with the executable
notebook so its own instructions stay accurate. Optional notebook execution
checks use the existing docs group for nbclient/nbformat.

Validation after authorization: exact trial equality, fresh-kernel notebook
execution outside checkout, ruff and scripts/check.sh --fast; commit and use
independent-review, then stop.

## Docs to-do

A subsequent docs step should replace the checkout-only instructions in the
page, notebook prose and README with standalone notebook instructions.
Untracked nvim.log was present at step start and is not part of this work.
