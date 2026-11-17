# PH.PERF.1 — Investigate current PH likelihood performance

Step starting commit: 54e3fa9f280f63bcd9c08865fc99e128cd69c370
Feature: issue [#8](https://github.com/vanamsterdam/survjax/issues/8), investigation.
Branch/worktree base: 54e3fa9f280f63bcd9c08865fc99e128cd69c370.
Previous completed step at this base: [TOOLING.FORMAT.1](../TOOLING.FORMAT.1/PLAN.md),
85f8275 (integrated by 54e3fa9). Performance dependencies:
[B4.2](../B4.2/PLAN.md), [B4.3](../B4.3/PLAN.md), and
[B3.5](../B3.5/PLAN.md). Existing completed feature selection in
scripts/spec-features.txt stays unchanged.

## Scope and implementation

Create scripts/benchmark_ph.py (under 300 implementation lines) and save its
float32/float64 CPU measurements in this step directory. No library, API,
dependency, protected-path, tolerance or docs changes. Spec coverage: 2, 3.2,
4.4, 7.1–7.3, 12.1 and 12.3. Spec 4.4 explicitly keeps likelihood/direct
methods on the PH formulas; equivalent Weibull is a benchmark control only.

Benchmark shared log_shape/log_scale and a binary covariate coefficient at
N=300 and 30,000, with prepared data passed dynamically to compiled functions.
Cover exact, exact/right, all four observation kinds, and all four kinds with
left/right/two-sided truncation. Compare PH-Weibull with the equivalent natural
Weibull in a moderate representable domain; add an unchecked hand formula for
exact/right only. Time summed objective and grad separately (matching issue #1's
grad measurement), plus direct survival/density components on exact/right data.
Separate placement/construction, lowering, compilation, first execution and
synchronized interleaved warmed execution; report versions, hardware, flags,
seed, parameter points and raw timing samples. No timing threshold tests.

Embed correctness checks against independently differentiated NumPy formulas
for likelihood values and all three log-parameter derivatives at two parameter
points, using existing implementation tolerances. These checks exercise the
benchmark, not new library behavior. Preserve all existing correctness tests;
run scripts/check.sh --fast before handoff. Write REVIEW.md, commit the step,
run independent-review, follow the optional diff-display hook, then stop.

## Decision and remaining sub-steps

Use the measurements to recommend a separate PH.PERF.2 implementation step
only if worthwhile and safe. Investigate whether generic PH tail recovery,
repeated survival evaluation and conditional interval defaults dominate.
Do not use a PH-equivalent likelihood shortcut: equivalent parameters can be
unrepresentable and delegation contradicts spec 4.4. Any future optimization
must preserve the PH formulas, numerical tails, invalid-row isolation and
first/second parameter derivatives in the specified regime. Implementation
and before/after correctness/performance evidence remain after this step.

## Docs to-do

None for public documentation in this investigation. Keep reproducibility,
comparison with issue #1's original measurements, limitations and next-step
recommendation in this step's REVIEW.md.
