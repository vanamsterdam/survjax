# SPEC63.NUMERICS-DOCS.1 — Document revision-63 tail accuracy

Step starting commit: 3701148319c988f29b3caa76425f68b5f3e9c004

Feature: numerical-behaviour documentation follow-up from
[SPEC63.LOG1MEXP.1](../SPEC63.LOG1MEXP.1/PLAN.md), final 3701148
(implementation 69ffc7c, independent review merge). Branch fastlane;
original worktree base 3852184bb0e6f0a0a9b746ad8b92a95f675b23d0.
Dependencies: spec revision 63 at 9c052c2 and the reviewed single-branch
log1mexp implementation. Completed feature selection is unchanged (all five
committed features); this step activates no feature.

Change only docs/numerical-behaviour.md and this step's records. Keep the
existing lower-tail example and the section anchor. Replace the superseded
claim of relative upper-tail log-CDF value precision with ordinary absolute
accuracy, possible rounding to zero when survival is below dtype epsilon,
and log_survival as the representation of the upper tail. Retain relative
first/second parameter derivative accuracy on guaranteed smooth interiors.
Update the late-time executable example to show zero log CDF, positive
survival and finite log survival; show a nonzero parameter derivative.
Explain that rounded log probability zero does not establish uninformative
coverage. Correct the earlier general precision sentence to make this
probability-near-one exception explicit.

Scope: one documentation topic, under 300 changed implementation lines.
No public API, module, dependency, source/test logic or protected path changes.
No new unit tests: this is prose and executable documentation, backed by the
preceding step's mpmath/derivative suites. Spec coverage: 4.1–4.2, 7.1–7.3,
12.1. Run scripts/build_docs.sh directly (both dtypes, warnings as errors),
inspect the rendered page and executed outputs in each dtype, and run the
required fast gate before handoff. Do not change files while the build's
identity check is in progress.

Commit the page and handoff; run independent-review; display the committed
step diff with the workspace hook, then stop. Do not edit the root roadmap.

## Docs to-do covered

This step covers the single numerical-behaviour topic listed in
SPEC63.LOG1MEXP.1: lower/upper-tail accuracy, late-time example, parameter
derivatives and use of log_survival. No additional docs topic is planned here.
