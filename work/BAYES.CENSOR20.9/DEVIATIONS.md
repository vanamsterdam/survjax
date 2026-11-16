# BAYES.CENSOR20.9 deviations

## D-BAYES.CENSOR20.9-001

- Step: BAYES.CENSOR20.9
- Spec section: 11.3; use-cases/02-bayesian-survival-details.md Data.
- What the spec says: c is the calibrated value for 55% censoring.
- What is implemented: fixed c = 20.0 for the Bayesian trial, without changing
  the shared generator or other demos. Regenerate trial/posterior evidence.
- Maintainer direction: explicitly requested 20.0 before merging, following
  the scope report identifying this requirement and regeneration work.
- Status: open (maintainer resolution required).
