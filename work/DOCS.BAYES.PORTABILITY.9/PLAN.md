# DOCS.BAYES.PORTABILITY.9 — Standalone notebook instructions

Step starting commit: 7a116527297b850096e99465b658ba66837a1a89
Worktree base: 54e3fa9f280f63bcd9c08865fc99e128cd69c370.
Branch: bayesian-use-case. Previous step:
[BAYES.PORTABILITY.9](../BAYES.PORTABILITY.9/PLAN.md).
Dependencies: that step's standalone notebook and successful outside-checkout
execution. Completed feature selection: unchanged.

One docs topic, under 100 changed documentation lines: update
docs/use-cases/bayesian-rmst.md, examples/bayesian_rmst/README.md and the
notebook's Markdown instructions. Remove checkout-only claims for the
notebook, distinguish the checkout-based runner and full-model snippet,
explain the notebook-local trial with a recorded calibration constant and
exact-equality check, and name the checked dependency versions for standalone
use (previous review nit N-2). No notebook code, tests, library API, dependency,
spec or protected-path changes. Previous code-step nit N-1 (unused fixture)
is outside this docs step and nonblocking.

Spec coverage: 11.3 and use-case 02, with the maintainer-approved local-draw
exception recorded by the preceding code step. No new deviation needed.
Validation: scripts/build_docs.sh directly (both dtypes), inspect rendered
instructions/download, scripts/check.sh --fast, and independent-review after
commit. No new tests for prose changes. Preserve the diff hook's untracked
nvim.log outside the checkout before review.

## Docs to-do covered

Covers the preceding code step's page/README portability instructions and
aligns notebook prose. No further documentation sub-step planned for issue #9.
