# PH.PERF.3 — NumPyro censored-Weibull comparison

Independent review: round 1/3

Step starting commit / previous step end: 2ab4bc99b86ff42e5ae2d996910707294ea1efcc
([PH.PERF.2](../PH.PERF.2/REVIEW.md)). Worktree base:
54e3fa9f280f63bcd9c08865fc99e128cd69c370. No feature selection changes.

## Read first

1. [PLAN.md](PLAN.md).
2. scripts/benchmark_ph.py: lazy NumPyro route, exact/right-only opt-in,
   version reporting and command reproduction. Measurement and oracle unchanged.
3. [float32.json](float32.json) and [float64.json](float64.json).

## Scope and invariants

The PH.PERF.2 library optimization is baseline-independent: it shares PH
survival for any baseline when the concrete wrapper and static observation
metadata qualify. It was independently tested over all four built-ins and
a custom baseline. Performance evidence so far is specifically PH-Weibull.
This step changes no library, dependency declaration, public API, docs, tests
or protected paths. NumPyro 0.22.0 already belongs to the examples group.

--numpyro adds RightCensoredDistribution with Weibull parameters
scale=b*exp(-eta/k), concentration=k, censored=~data.exact. It scores data.lower.
The control is used only for exact and exact/right likelihoods; no NumPyro
comparison for other censoring, truncation or direct-method components.
Lazy imports preserve the script's default dependency footprint.

The installed NumPyro log_prob uses log1p(-clip(cdf,0,1-eps)) for censored
values and baseline.log_prob for exact values. Moderate times/parameters here
stay away from the clipping threshold. This measures the actual NumPyro API;
it is not an extreme-tail equivalence claim or a replacement likelihood.
NumPyro's parameter transformation is included, just like equivalent Weibull.

The existing independent NumPy analytical oracle checks summed objectives
and all three parameter gradients at two parameter points for each route.
No timing assertions, new pytest tests, changed tolerances or duplicated
likelihood reference code. JAX compilation caching is disabled; data and arm
remain dynamic arguments. Every timed result is synchronized. Lowering,
compilation, first execution and steady-state samples are separate fields.
Order rotates across the same-run controls, ten warmups and 200 samples each.

## Reproduction and results

```sh
JAX_ENABLE_X64=0 JAX_PLATFORMS=cpu uv run --group examples python scripts/benchmark_ph.py --numpyro --sizes 30000 --warmups 10 --repeats 200 --output work/PH.PERF.3/float32.json
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run --group examples python scripts/benchmark_ph.py --numpyro --sizes 30000 --warmups 10 --repeats 200 --output work/PH.PERF.3/float64.json
```

Sequential dtype processes before the fast gate. Same host and fixed-seed
workload as preceding steps: macOS 26.7.2 arm64, 16 logical CPUs, Python 3.12.9,
JAX/jaxlib 0.11.2, NumPy 2.5.3, NumPyro 0.22.0, survjax 0.0.1a3, CPU backend.
Restricted CPU-brand lookup falls back to processor=arm; the earlier runs
identified this workspace host as Apple M4 Max. Both JSON revisions identify
the step starting commit and clean library source; the benchmark edits are
committed in this step. Peak memory is not measured.

N=30,000 exact/right, median objective / gradient microseconds:

| route | float32 | float64 |
|---|---:|---:|
| optimized PH | 180.9 / 633.5 | 308.5 / 1336.7 |
| generic PH control | 233.5 / 719.8 | 396.2 / 1408.6 |
| equivalent Weibull | 133.7 / 275.5 | 215.6 / 446.3 |
| NumPyro censored Weibull | 89.7 / 288.1 | 358.4 / 513.7 |
| unchecked formula | 72.5 / 80.4 | 125.6 / 235.3 |

Lowering+compilation for those gradients, milliseconds (single observation
per executable, not a statistically replicated compilation comparison):

| route | float32 | float64 |
|---|---:|---:|
| optimized PH | 147.1 | 238.5 |
| generic PH control | 239.6 | 424.8 |
| equivalent Weibull | 83.3 | 130.8 |
| NumPyro censored Weibull | 42.7 | 37.4 |
| unchecked formula | 22.1 | 21.1 |

The [original issue #1 table](https://github.com/vanamsterdam/survjax/issues/1)
reported float32 gradients of 74 us unchecked, 308 us plain Weibull, 516 us
NumPyro, 1277 us PH. Current NumPyro is 288 us and PH is 634 us. This run's
unchecked 80 us is close to that reported 74 us, but that does not establish
matching hardware/cohorts or a controlled historical speedup. The previous
PH.PERF.2 float32 confirmation was much noisier (PH 2264 us, unchecked 273 us).
Use same-run comparisons: optimized PH is 2.20x NumPyro in float32 and 2.60x
in float64. PH retains an overhead; the new control completes the internal
comparison without changing the production optimization.

## Validation

- Focused Ruff format/lint: passed.
- Both commands above: exit 0, 44 records per dtype. Every objective/gradient
  check passed at both parameter points, including exact-only and exact/right
  NumPyro controls. No mode changes within a process.
- scripts/check.sh --fast: exit 0, all gates passed. Guard clean; Ruff format
  (308 files), lint and mypy (25 files) passed. Implementation float64:
  1058 passed, 8 skipped (7.68s); float32: 1060 passed, 6 skipped (7.59s).
  Source doctests: 4 passed per mode. Acceptance float64: 4417 passed,
  46 skipped (26.54s); float32: 4384 passed, 69 skipped (28.75s), each with
  the two existing PH reference-arithmetic warnings. git diff --check clean.
- No slow-test refresh needed: no pytest tests or library behavior changed.
  No full nonfast gate, docs build or R fixture regeneration in this step.

## Remaining

No further sub-steps for this comparison. Other baselines' performance remains
unmeasured; NumPyro comparisons here do not cover interval/left censoring,
truncation or extreme tails. Stop after independent review; no publishing.
