# SPEC63.LOG1MEXP.1 — Single-branch log1mexp

Step starting commit: c4023cf392a8fb71feb5187af15e396514c244b9

Feature: implement the optional revision-63 simplification authorized in the
maintainer's original task. Previous step:
[SPEC62-63.1](../SPEC62-63.1/REVIEW.md), final c4023cf. Branch fastlane;
original worktree base 3852184bb0e6f0a0a9b746ad8b92a95f675b23d0;
resumed on maintainer merge 29a91a7. Dependencies: spec revision 63 at
9c052c2 and all five committed features; completed feature selection unchanged.

Change src/survjax/distributions/_log_math.py to compute safe_log1mexp using
log(-expm1(x)), substituting a stopped finite operand at zero before operating.
Retain the existing endpoint and invalid-value behavior. No new public API,
runtime module, dependency or clipping.

Update tests/unit/test_log_cdf_tails.py's old nonzero upper-tail value
requirement to spec 12.1's ordinary absolute value accuracy. Keep the existing
zero-absolute-tolerance derivative/Hessian checks and all lower-tail checks.
Update the explanatory comment in tests/impl_tolerances.py, without changing
any tolerance values. Add tests/unit/test_log1mexp.py for endpoint/invalid
values, mpmath values, and relative first/second derivatives of mixed vectors
containing zero, interiors, saturated upper tails and negative infinity.

Run touched tests in full in both dtype modes, plus A4/A5 log-CDF, A11 direct
methods and observation-censoring numerics acceptance tests. Run the full and
fast gate in both dtypes before handoff. Refresh the slow-test inventory only
if a new test exceeds one second. Target under 300 implementation changed lines.
Spec coverage: 4.1–4.2, 4.5, 7.1–7.3, 12.1. Commit records and code together,
run independent-review, display the diff and stop after this step.

Before this step the maintainer explicitly authorized discarding the throwaway
reviewer checkout's stale files. An unrestricted checkout --detach --force at
c4023cf and clean -fd limited to tests/spec restored it; status is clean,
SOURCE_COMMIT matches a0e7f49 and the Slice B fixture matches byte for byte.
No reviewer role, settings or prompt was changed. Future reviews retain the
skill's normal procedure; this one-off repair does not change that workflow.

The plain formula failed upper-tail derivatives in JAX 0.11.2 in both dtypes:
its default expm1 autodiff reconstructs exp(x) as expm1(x)+1. Inspection of
the installed JAX code and direct both-dtype derivative probes identify the
native solution: lax.expm1(..., accuracy=lax.AccuracyMode.HIGHEST) uses exp(x)
in its derivative. Use that primitive; no custom derivative is necessary.
The existing JAX minimum is 0.11.2, where this mode is available. This supplies
the existing derivative contract without clipping, new API or dependencies.
Update the now-inaccurate log_cdf docstrings in _survival_methods.py,
exponential.py and weibull.py with the numerical change.
The broader retained suite also has superseded relative value assertions in
tests/unit/test_proportional_hazards_tails.py and test_family_infinity.py.
Update only their log-CDF value comparisons to ordinary accuracy; retain
relative derivative checks and add an infinity-hook log-cure derivative case.
The extended cure-fraction test takes 1.43s, so run the required measured
slow-test inventory refresh and include its generated list with this step.
The refresh changes 393 generated lines, exceeding the usual step-size target
alongside about 170 implementation/test lines. The maintainer explicitly
approved including that generated refresh in this same step.

## Docs to-do

Follow-up docs step: update docs/numerical-behaviour.md's "Precision in both
tails" explanation and late-time example to revision 63: absolute accuracy
for upper-tail log F, relative derivative accuracy, and log_survival for the
tail itself. Do not edit docs in this code step. Public signatures and shapes
remain unchanged.
