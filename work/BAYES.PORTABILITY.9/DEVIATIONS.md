# BAYES.PORTABILITY.9 deviations

## D-BAYES.PORTABILITY.9-001

- Step: BAYES.PORTABILITY.9
- Spec section: 11.3; use-cases/02-bayesian-survival.md and details.
- What the spec says: the trial comes from the shared generator; exact values
  and keys belong to the use-case notes.
- Proposal: make the notebook standalone by spelling out its small synthetic
  draw calculation locally, using the same key and recorded calibrated
  censoring limit. Preserve the runner's shared generator and verify exact
  trial equality in an optional example check. This duplicates a small draw
  calculation to remove the runtime dependency on repository example modules.
- Maintainer decision: approved in this conversation before implementation.
- Status: open
