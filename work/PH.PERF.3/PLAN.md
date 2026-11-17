# PH.PERF.3 — Complete the internal NumPyro comparison

Step starting commit: 2ab4bc99b86ff42e5ae2d996910707294ea1efcc
Previous step: [PH.PERF.2](../PH.PERF.2/REVIEW.md), ending at that commit.
Branch/worktree base: 54e3fa9f280f63bcd9c08865fc99e128cd69c370.
Feature: issue #8 benchmark comparison; completed feature selection unchanged.

Extend scripts/benchmark_ph.py with an opt-in --numpyro control. Use the
already locked examples-group NumPyro 0.22.0, imported lazily only for this
control; no dependency, library, API, protected-path or docs changes. Compare
RightCensoredDistribution(Weibull(scale=b*exp(-eta/k), concentration=k),
censored=~data.exact) on moderate exact/right inputs only. Keep all existing
controls in the same synchronized, rotating-order run. Record NumPyro's
version and the examples-group invocation in the environment header.

Existing independent analytical checks must validate objective and all three
parameter gradients at two points for every route. No new pytest test or
timing threshold: the executable benchmark checks its new control itself.
Run N=30,000, ten warmups, 200 repeats in separate dtype processes; save raw
measurements here and compare the original table with within-run controls.
NumPyro's installed implementation clips its CDF at 1-eps for censored values;
this moderate-workload comparison makes no extreme-tail equivalence claim.
Spec context: 4.4, 7.2 and 12.3. Target under 50 implementation changed lines.
Run scripts/check.sh --fast, write REVIEW.md, commit, independently review,
display the step diff and stop. No additional sub-steps or public docs needed.
