# PH.PERF.2 — Shared PH survival in exact/right likelihoods

Independent review: round 1/3

Step starting commit / previous step end: 26f6b179c8740a28af03a49e9d9c9cb3068c9c87
([PH.PERF.1](../PH.PERF.1/REVIEW.md)). Branch/worktree base:
54e3fa9f280f63bcd9c08865fc99e128cd69c370. Existing feature selection is unchanged.
This is the implementation sub-step of issue #8. No API or docs changes.

## Read first

1. [PLAN.md](PLAN.md); spec 4.4 and 7.1–7.3.
2. src/survjax/likelihood.py: the short concrete-PH, untruncated exact/right
   guard and early return; parameter preparation and validation below remain
   unchanged. The generic evaluator after the early return is unchanged.
3. tests/unit/test_ph_likelihood_shared_survival.py: independent mpmath
   reference, all four built-in baselines' joint derivatives, numerical safety
   and type/metadata guards. Existing isolation and tail tests supplement it.
4. scripts/benchmark_ph.py: inherited UnsharedPH control and route selection;
   measurement logic and independent analytical reference are unchanged.
5. Saved measurements here and [PH.PERF.1](../PH.PERF.1/REVIEW.md).

## Invariants and correctness evidence

The guard requires concrete type ProportionalHazards, possible exact AND right
rows, and proof of absence of left/interval censoring and both truncation modes.
This retains arbitrary subclass overrides and all non-PH behavior. Static
single-kind and empty batches retain the previous pruning. Conservative
metadata can select the new path even if all current rows happen to be one
kind; eligibility never depends on inspecting dynamic values.

Spec 4.4 gives PH log f=log h+log S. With no truncation, exact rows need those
two terms and right rows need log S alone. Shared survival evaluates each valid
row's lower endpoint once. Its bad-row substitute is zero. Hazard evaluation
uses active exact times or the same stopped positive interior reference used
by the old density branch. Select zero for inactive hazard contributions
before adding survival. Actual parameters and active times are never stopped.
Data/current-field checks and invalid-parameter gradient preparation precede
both routes; bad rows still return NaN. Common-dtype promotion and trailing-axis
broadcasting are unchanged. No closure delegation, new numerical primitive,
clipping, floors, callbacks, host transfers or custom differentiation rules.

- test_shared_values_and_joint_derivatives_match_independent_law checks
  spec 4.4 values and 7.3 baseline/eta Jacobians and all four forward/reverse
  Hessian compositions for Exponential, Weibull, LogNormal and LogLogistic,
  two moderate parameter points, eager and jit(vmap). The oracle uses 80-digit
  mpmath baseline density/survival definitions and differentiates the PH law,
  never library outputs. Zero-time censoring has exactly zero values and
  parameter derivatives. Valid eager values/gradients run under debug_nans.
- test_inactive_generic_hazard_uses_interior_time uses a custom sqrt-time
  baseline whose generic hazard cannot be differentiated at zero. The exact
  log density at one has independent algebra log(r/2)+eta-r*exp(eta), and the
  right-at-zero row is constant zero. Values/Jacobians/Hessians compare under
  debug_nans using existing tolerances.
- test_extreme_scaling_keeps_ph_values_without_equivalent_parameters uses
  100-digit mpmath values and zero absolute tail tolerance: positive eta
  recovers a baseline hazard rounded to zero, negative eta keeps tiny normal
  PH hazards despite an overflowing equivalent scale. Extreme baseline
  parameter derivatives are not claimed; spec 7.3's exception is unchanged.
- test_dynamic_observations_reuse_trace_and_keep_validation covers raw-input
  issue propagation, current-window corruption with clean stored issues, and
  one trace across changed time/event values (spec 3.2/7.1–7.2).
- test_disqualifying_modes_retain_density_hook independently checks each
  censoring/truncation exclusion. test_shared_route_uses_one_survival_and_
  preserves_subclass_density checks one survival call and subclass hook use.
- Existing test_proportional_hazards_isolation.py checks values, diagnostics,
  invalid neighboring baseline/eta entries and cross Hessians, plus compile
  reuse across valid/invalid dynamic parameters. Its mixed exact/right rows
  now exercise the new path, including a custom parameter condition.
- Existing PH tails retain direct-method values and derivatives. Existing
  test_wrapper_censoring_derivatives.py independently covers all four baselines,
  wrappers and observation kinds across no/left/right/two-sided truncation,
  including Hessians. Existing test_likelihood_and_second_order_interior_
  derivatives supplies order-two finite differences in both AD modes on the
  newly selected path.

## Measurements and decision

Environment and fixed-seed workload are the same as PH.PERF.1; see each JSON
environment header and the prior report. Run sequentially per dtype:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python scripts/benchmark_ph.py --output work/PH.PERF.2/float64.json
JAX_ENABLE_X64=0 JAX_PLATFORMS=cpu uv run python scripts/benchmark_ph.py --output work/PH.PERF.2/float32.json
```

The added ph_generic control inherits PH's primitives, fields and validation
unchanged. Registration reapplies the same direct-method policy to inherited
raw methods; only concrete type differs. It fails the new type guard and uses
exactly the unchanged generic evaluation arithmetic of the previous step.
Both executables receive identical dynamic data/arm/parameters, and the
existing independent oracle checks every route's objective and gradient at
two points. All other baselines and measurement mechanics remain unchanged.
There are 80 records per dtype, with synchronized interleaved warmed samples
and separate lowering/compilation/first execution. No timing thresholds.

The JSON revision names the step starting commit. Source was intentionally
modified at measurement time (source_status records this); the candidate is
committed in this step, not the starting tree. After the initial runs, the
benchmark gained a CPU-brand lookup fallback; timing/numerical code is identical.
The inherited control supplies a within-run old/new comparison; different-day
absolute latency comparisons to PH.PERF.1 are not controlled experiments.

First-run N=30,000 exact/right median objective / gradient microseconds:

| dtype | shared PH | generic PH control |
|---|---:|---:|
| float64 | 349.5 / 1469.3 | 437.3 / 1588.7 |
| float32 | 727.9 / 1686.1 | 900.2 / 2162.9 |

Float64 lowering+compilation for that gradient is 248.3 ms shared versus
455.3 ms generic. The first float32 run is much noisier than PH.PERF.1;
even unchanged exact-only controls differ. Confirmation runs therefore use
10 warmups and 200 interleaved samples, N=30,000 only, in separate processes
after the full test runs, with no implementation changes:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python scripts/benchmark_ph.py --sizes 30000 --warmups 10 --repeats 200 --output work/PH.PERF.2/float64-confirm.json
JAX_ENABLE_X64=0 JAX_PLATFORMS=cpu uv run python scripts/benchmark_ph.py --sizes 30000 --warmups 10 --repeats 200 --output work/PH.PERF.2/float32-confirm.json
```

Confirmation median microseconds for exact/right, shared / generic control:

| dtype | objective | gradient | lower+compile gradient (ms) |
|---|---:|---:|---:|
| float64 | 302.7 / 405.1 | 1323.4 / 1431.0 | 257.1 / 426.9 |
| float32 | 905.1 / 1278.4 | 2264.2 / 3063.3 | 369.4 / 926.0 |

Objective reductions are 25.3%/29.2%, gradient reductions 7.5%/26.1%
(float64/float32). These are local ratios of medians, not guarantees. Float64
has shared faster in 200/200 objective pairs and 180/200 gradient pairs;
paired generic/shared gradient ratio p10/median/p90 is 1.001/1.088/1.163.
Float32 remains noisy despite the longer run: 156/200 objective and 145/200
gradient pairs favor shared; gradient paired ratios are .560/1.378/2.719.
Do not interpret 26.1% as a precise portable speedup. Both first and confirmation
runs favor the shared path, with a repeatable modest float64 gradient gain.

Unchanged-path controls help distinguish that gain from class/timing artifacts:
float64 exact-only gradients are 1496.0/1499.9 us, mixed 5741.9/5724.8 us,
truncated 5795.9/5819.5 us. Float32 is 2400.6/2500.1, 8533.9/8325.8 and
12911.4/12574.9 us respectively. No improvement is claimed for these paths.
Equivalent Weibull still has cheaper gradients (443.4 us float64, 1208.9 us
float32); unchecked formulas lack the required validation and numerical guards.
The full PH/Weibull and PH/unchecked gaps are not eliminated.

Decision: retain the short shared-survival path. It removes redundant PH work,
reduces objective and compilation costs, and gives a measured gradient benefit
without weakening the tail recovery or validity checks. No new fast formula or
representable-equivalent-parameter assumption is introduced. This completes
issue #8's measured optimization/effect/correctness scope locally, building on
PH.PERF.1's baseline and comparison with the original report. Integration and
independent review remain; further interval specialization is optional work.

Compiler estimates for the same float64 N=30,000 gradient are saved in
compiler_costs.json. Obtained by compiling jax.grad of benchmark_ph._objective
with each route and reading executable.cost_analysis(): transcendental counts
660,004 shared versus 1,170,004 generic; estimated FLOPs 42,870,168 versus
78,060,272; estimated bytes accessed 53,370,608 versus 95,460,896. These are
compiler cost estimates, not measured instruction counts, memory use or a
prediction of runtime speedup. They confirm the removal of repeated work.

## Validation

- Focused Ruff formatting/lint and mypy (25 source files) passed. Implementation
  diff is about 300 lines: 18 source additions, 270-line new test file, 25
  changed benchmark lines (including metadata fallback for PH.PERF.1 nit N-1).
- Restricted-environment N=16 float32 benchmark smoke after that fallback:
  exit 0; all 40 records passed independent checks (1 warmup, 2 repeats).
  Smoke timings are not performance evidence.
- Initial nine-test focused runs: one failed / eight passed per dtype. The
  negative-eta fixture expected a subnormal log survival that CPU exp flushes
  to zero. Explicit old/new checks showed the same zero in the generic path:
  eta=-720 (x64) or -100 (x32), k=.125, b=t=1. Replaced the stress parameter
  with eta=-600/-60, keeping H normal while equivalent scale still overflows.
  This tests the intended closure-overflow property without changing any
  tolerance, library behavior or subnormal guarantee.
- JAX_ENABLE_X64=1 uv run --group test pytest -n 6
  tests/unit/test_ph_likelihood_shared_survival.py
  tests/unit/test_proportional_hazards_isolation.py
  tests/unit/test_proportional_hazards_tails.py
  tests/unit/test_wrapper_censoring_derivatives.py --durations=12:
  114 passed in 298.98s. Four metadata-guard cases were then added.
- Same command with JAX_ENABLE_X64=0 after those additions:
  118 passed in 284.42s.
- Float64 complete new file plus existing
  test_proportional_hazards.py::test_likelihood_and_second_order_interior_derivatives:
  14 passed in 32.76s, including the four added guard cases and check_grads.
- Both full benchmark commands: exit 0, all 80 records per dtype passed
  independent value/gradient checks at two parameter points.
- Both longer confirmation commands: exit 0, all 40 records per dtype passed
  independent value/gradient checks at two parameter points.
- PYTEST_WORKERS=6 scripts/update_slow_tests.sh: exit 0; float64
  implementation 1540 passed, 8 skipped (667.03s), acceptance 4863 passed,
  47 skipped, 2 existing PH reference-arithmetic warnings (336.92s).
  Float32 implementation: 1539 passed, 9 skipped (602.64s); acceptance:
  4774 passed, 126 skipped, 2 existing PH warnings (154.96s).
  The script ignores pytest failures, so summaries are captured independently
  in /private/tmp/survjax-PH.PERF.2-suite-logs rather than inferred from exit 0.
  It wrote 937 slow classifications, including 11 of the 13 new cases. The
  mechanical list diff is 478 additions/63 deletions and is isolated from
  handwritten implementation in its own commit, as in B3.5.
- scripts/check.sh --fast: exit 0, all gates passed. Guard clean; Ruff format
  (307 files), lint and mypy (25 source files) passed. Implementation float64:
  1058 passed, 8 skipped (23.64s); float32: 1060 passed, 6 skipped (19.22s).
  Source doctests: 4 passed per mode. Acceptance float64: 4416 passed,
  47 skipped, 2 existing PH warnings (66.34s); float32: 4382 passed,
  71 skipped, 2 existing PH warnings (57.09s). Counts differ from PH.PERF.1
  because classification was regenerated; full-suite counts are above.
- git diff --check: clean. No full nonfast check.sh invocation, docs build or R
  regeneration; complete dtype-mode pytest suites were run by the refresh.

## Remaining scope

General mixed/truncated PH likelihoods still use the existing interval
defaults. They are correctness controls here, not optimized paths. No claim
of eliminating the PH/Weibull or PH/unchecked gap, or of stabilizing extreme
parameter derivatives. No further implementation sub-step is required for the
focused issue #8 scope; independent review and integration remain. Further PH
tail/interval performance work would be a separately assigned step.
