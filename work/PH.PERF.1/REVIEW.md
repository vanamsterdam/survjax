# PH.PERF.1 — Current PH likelihood performance

Independent review: round 1/3

Step starting commit / worktree base: 54e3fa9f280f63bcd9c08865fc99e128cd69c370.
Previous completed step: TOOLING.FORMAT.1, 85f8275500afc3746643a6c7eb3ace0edbc8cfc0
(integrated by the starting commit). This is investigation sub-step 1 of issue
[#8](https://github.com/vanamsterdam/survjax/issues/8); no production optimization
or issue-completion claim. Existing completed feature selection is unchanged.

## Read first

1. [PLAN.md](PLAN.md); spec 4.4, 7.1–7.3 and 12.3.
2. scripts/benchmark_ph.py: observation generation, independent analytical
   reference, three controls, separate compilation and interleaved timing.
3. [float32.json](float32.json), [float64.json](float64.json): environment plus
   64 records per dtype with raw samples, flags and objective/gradient results.
4. Existing src/survjax/distributions/wrappers.py and likelihood.py for the cost
   investigation below. Neither file changes in this step.

## Method and correctness invariants

Apple M4 Max, 16 logical CPUs, macOS 26.7.2 arm64, Python 3.12.9,
JAX/jaxlib 0.11.2, NumPy 2.5.3, survjax 0.0.1a3, CpuDevice(0).
Library source is clean at the starting commit in both saved runs.
Each dtype runs in its own process, sequentially, before the correctness gate.
No dependency, library, API, tolerance, protected-path or public-docs changes.

Reproduce from the repository root:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python scripts/benchmark_ph.py --output work/PH.PERF.1/float64.json
JAX_ENABLE_X64=0 JAX_PLATFORMS=cpu uv run python scripts/benchmark_ph.py --output work/PH.PERF.1/float32.json
```

Seed 20261116. Times uniform in [0.75,2.5), binary arm; parameters
(log(1.3),log(2),0.25). Exact/right has approximately 65% exact observations.
Mixed has equal exact/right/left/interval counts. Truncated crosses those four
kinds with windows (0,inf], (.5,inf], (0,3], (.5,3]. Interval width is .5;
all active times and censoring sets intersect valid positive-mass windows.
Concrete metadata is recorded; dynamic prepared data and arm are executable
arguments, never captured constants. Construction inside jit, per-observation
free parameters, other baseline families and accelerators are outside this run.

Objectives sum pointwise contributions. Each route times objective and grad
separately, matching the original report's grad measurement. Three warmups
precede 30 samples per executable; order rotates across the routes/derivative
modes in each workload. Every output is synchronized. Generation, placement
and construction are combined in the preparation field, excluded from timings.
Lowering, compilation and first execution are separately synchronized/timed;
persistent compilation caching is explicitly disabled. Peak memory is not
measured. Compilation order is fixed, so cold start effects remain possible.

Embedded checks compare every likelihood route at two parameter points against
independently derived NumPy values and analytical derivatives:
H(t)=exp(gamma*arm+k*(log(t)-log(b))), dH=H*(k*log(t/b),-k,arm).
Density uses log(k)-log(t)+log(H)-H; interval mass uses
-H(a)+log(1-exp(-(H(b)-H(a)))), then subtracts log window mass.
Endpoint limits are substituted analytically. These moderate-domain formulas
are benchmark oracles, not general numerical replacements. Existing ATOL/RTOL
are loaded from tests/impl_tolerances.py, unchanged. Direct-method component
checks instead compare with AD of an unchecked independent defining formula.
No timing assertions or new pytest tests; the benchmark checks its own inputs
and references. No new library behavior needs correctness tests in this step.

## Measurements

Median warmed microseconds at N=30,000; objective / parameter gradient:

| dtype | mix | PH | equivalent Weibull | unchecked formula |
|---|---|---:|---:|---:|
| float32 | exact | 172.3 / 655.3 | 111.0 / 237.9 | 60.8 / 71.3 |
| float32 | exact/right | 219.4 / 685.3 | 117.6 / 256.5 | 64.9 / 71.7 |
| float32 | mixed | 748.3 / 2916.5 | 416.9 / 1192.6 | unavailable |
| float32 | truncated | 1071.9 / 4014.0 | 614.4 / 1327.7 | unavailable |
| float64 | exact | 308.9 / 1321.4 | 196.1 / 419.4 | 126.2 / 222.4 |
| float64 | exact/right | 389.8 / 1367.6 | 219.2 / 449.5 | 119.1 / 220.6 |
| float64 | mixed | 1381.1 / 5643.4 | 684.8 / 1807.6 | unavailable |
| float64 | truncated | 1390.8 / 5720.5 | 701.9 / 1755.8 | unavailable |

At N=300, exact/right PH gradients are 73.5/87.0 us in float32/float64;
equivalent Weibull 40.6/57.7 us; unchecked 13.2/18.4 us. All smaller workloads
and individual samples are saved. No fixed speed guarantee: these are local
latencies including dispatch, not isolated device kernel timings.

For N=30,000 exact/right gradients, lowering / compilation / first execution
in milliseconds (warmed samples exclude all three):

| dtype | route | lowering | compilation | first execution |
|---|---|---:|---:|---:|
| float32 | PH | 62.4 | 186.3 | 1.4 |
| float32 | equivalent | 23.4 | 62.0 | 1.0 |
| float32 | unchecked | 5.1 | 19.1 | 0.7 |
| float64 | PH | 58.5 | 359.2 | 2.7 |
| float64 | equivalent | 24.8 | 108.7 | 1.8 |
| float64 | unchecked | 4.6 | 15.7 | 1.3 |

Issue [#1, finding 6](https://github.com/vanamsterdam/survjax/issues/1) at
3abefc5 reported float32 PH 1277 us versus unchecked 74 us (17.3x), plain
Weibull 308 us, NumPyro 516 us; compile 805/80/322/216 ms respectively.
Here PH/unchecked is 9.6x and PH/equivalent 2.7x in float32 (6.2x/3.0x in
float64). Current float32 PH latency is about 54% of the reported old latency;
unchecked is similar at 71.7 us. This is a comparison to reported numbers, not
a controlled old/new experiment: original hardware, exact data, event fraction,
and captured/dynamic argument details are unavailable. Equivalent Weibull here
also includes the parameter transformation, unlike a plain unrelated Weibull.
NumPyro is intentionally excluded: no adapter dependency or integration work.

## Cost investigation and recommendation

Source inspection confirms that each PH log_survival computes baseline
log_survival, a baseline log_cdf at safely substituted times, and the general
scaling ladder. That recovery is necessary when baseline survival rounds to
one but a large PH multiplier makes the cumulative hazard representable.
PH log_prob adds baseline log_hazard and calls PH log_survival again.
Untruncated exact/right likelihood already prunes interval and normalization
work, but still requests survival and density at distinct substituted times.
For mixed/truncated observations, PH inherits generic interval hooks; these
evaluate survival and CDF at both endpoints, with additional window evaluation.
Equivalent Weibull instead has stable specialized interval increment formulas.

Standalone N=30,000 PH survival/density gradient medians are 939/1536 us
float32 and 1149/1397 us float64. These are separate timing blocks, not
additive profiles: float32 component blocks even exceed the whole-likelihood
block, indicating drift/load sensitivity. The largest within-row max/min ratio
is 10.9 in float32 and 4.1 in float64. Use interleaved within-workload ratios
and source inspection to identify opportunities; do not infer exact cost
percentages from these component timings or call this a kernel profile.

The remaining roughly 2.5–3.3x large-batch gradient gap to the guarded Weibull
control justifies a separate implementation experiment. Recommend PH.PERF.2
first target duplicate PH survival work in the untruncated exact/right path,
retaining the same PH formula and safe substitutions. For example, combining
the active survival time across exact/right rows could share its calculation
while evaluating hazards at separately protected density times. Any such
change must be restricted so that arbitrary custom density overrides retain
their behavior. Runtime lower-tail pruning is another candidate, but inactive
zero-time rows and vmap execution prevent assuming recovery is never needed.
These are proposals, not measured candidate speedups or approved API changes.
General interval specialization can remain a later topic if still worthwhile.

Do not delegate to _ph_equivalent. Spec 4.4 prohibits that route for likelihoods
and direct methods. Independently checked counterexamples at exact t=1,
baseline Weibull(k=.125,b=1): eta=-100 (float32) gives PH likelihood -102.079445;
eta=-720 (float64) gives -722.07944154. The actual private closure returns
scale=inf in both cases, so its likelihood is NaN. PH values remain finite.
Both assertions passed in separate dtype processes; extreme parameter
derivatives are not claimed (spec 7.3).

## Validation

- Focused Ruff format/lint passed; 266-line benchmark implementation.
- N=16 float64 smoke: all 32 records passed (2 repeats, 1 warmup).
  The initial sandboxed smoke stopped before timing because CPU metadata
  access via sysctl was denied; rerunning with escalation passed.
- Full float64 and float32 benchmark commands above: exit 0, each 64 records
  with all value/gradient checks passing at both parameter points.
- Separate private-closure counterexample checks: exit 0 in both dtypes.
- scripts/check.sh --fast: exit 0, all gates passed. Guard clean; Ruff
  formatting (303 files), lint and mypy (25 source files) passed.
  Implementation float64: 1219 passed, 8 skipped (16.19s); float32:
  1221 passed, 6 skipped (18.57s). Source doctests: 4 passed per mode.
  Acceptance float64: 4653 passed, 47 skipped (52.56s); float32:
  4595 passed, 95 skipped (41.15s). Both acceptance runs report the two
  existing reference-arithmetic warnings in test_a11_ph_direct_methods.py.
  Full nonfast gate, docs build and R fixture regeneration were not run:
  this step adds only a standalone benchmark and its records.
- git diff --check: clean.

## Remaining sub-steps

PH.PERF.2: implement and measure a safe optimization, with focused boundary,
tail, invalid-parameter isolation, mixed censoring/truncation and first/second
parameter-derivative evidence plus before/after timings; alternatively record
why measured candidates are not worthwhile. This investigation supplies the
baseline and recommends proceeding. Stop after this step's independent review,
as required by the working agreement; issue #8 remains open.
