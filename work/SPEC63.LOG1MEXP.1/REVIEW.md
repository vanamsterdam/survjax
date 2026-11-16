# SPEC63.LOG1MEXP.1 — Single-branch log arithmetic

Independent review: round 1/3

## Read first

1. PLAN.md and spec 7.2 / 12.1, revision 63.
2. src/survjax/distributions/_log_math.py: safe operand at zero, one value
   formula, and native highest-accuracy expm1.
3. tests/unit/test_log1mexp.py: independent values and analytic derivatives.
4. tests/unit/test_log_cdf_tails.py: revised value contract; relative
   derivative/Hessian assertions remain unchanged.
5. Corrected log_cdf docstrings in _survival_methods.py, exponential.py and
   weibull.py; Docs to-do in PLAN.md.

## Change and invariants

safe_log1mexp now uses log(-expm1(x)) with a stopped finite operand substituted
at x=0 before evaluation. Endpoint and invalid-input behavior are unchanged.
The two-branch value calculation is removed. Upper-tail log F can round to
zero; structural full coverage is still determined from observation sets, not
from a zero contribution. No public API, static metadata, parameter dtype,
feature selection, dependency, floor, clipping or tolerance value changes.

Native lax.expm1 with AccuracyMode.HIGHEST is necessary here: installed JAX
0.11.2's default expm1 JVP uses expm1(x)+1, which loses upper-tail derivatives
once the value rounds to -1. The highest-accuracy JVP uses exp(x) directly.
JAX 0.11.2 is already the declared minimum. No custom derivative remains.
First and second derivatives preserve the existing relative-accuracy contract.

## Spec evidence

- 7.2: test_log1mexp_endpoints_and_invalid_values checks zero, negative infinity,
  positive and NaN inputs. Mixed derivative tests exercise safe substitution
  under vmap and jit with debug_nans active.
- 7.2/12.1: helper values use 100-digit mpmath; generic/built-in log-CDF values
  use mpmath across both tails and through grids. Only obsolete upper-tail
  value assertions adopt the spec's ordinary ATOL/RTOL. Existing TAIL_ATOL=0
  and TAIL_RTOL values are unchanged for gradients/Hessians and lower tails.
- 7.3/12.1: new mixed-vector tests use hand-derived
  f'=-exp(x)/(1-exp(x)) and f''=-exp(x)/(1-exp(x))^2 at x=-0.5, -33 and
  -600 (float64) / -70 (float32), including all four second-order AD
  compositions. Expected numbers are evaluated independently with mpmath.
- A4/A5 acceptance: generic/built-in upper-tail log F and underflowed-H lower
  tails retain values and first/second parameter derivatives.
- A11 direct-method acceptance: PH value policies and retained derivatives.
- observation.censoring numerics: left-censored upper tails, wide intervals,
  summed rows and Hessians. test_interval_defaults.py retains generic hook
  tail selection, precision-loss taxonomy and debug_nans coverage.

Focused command in each dtype (JAX_ENABLE_X64=1 and 0), with all five committed
features in SPEC_FEATURES:

`uv run --group test pytest -n 6 -q --durations=10 tests/unit/test_log1mexp.py tests/unit/test_log_cdf_tails.py tests/unit/test_interval_defaults.py tests/spec/test_a4_log_cdf_upper_tail.py tests/spec/test_a5_log_cdf_tails.py tests/spec/test_a11_ph_direct_methods.py tests/spec/test_observation_censoring_numerics.py`

Final native implementation: float64 262 passed; float32 255 passed / 3 skipped.
Each had two existing NumPy PH reference warnings. New-test durations in the
JUnit reports are <=0.655s (float64), <=0.674s (float32); no new test exceeds
one second. The later extended cure-fraction check takes 1.41s / 1.43s, so
the required slow-test inventory refresh is also part of this step.

The initial plain jnp.expm1 implementation failed 47 tests per dtype:
float64 215 passed / 47 failed; float32 208 passed / 3 skipped / 47 failed.
Failures exposed the default expm1 derivative rounding described above, in
independent spec tests as well as implementer checks. A temporary analytic JVP
passed but was removed in favor of JAX's native accuracy mode. All final
focused results above are from the native implementation. No failed spec test
was edited, skipped or xfailed; no tolerance value was loosened.

The first full gate then found three retained value assertions still imposing
the superseded relative-value requirement: tiny cure fraction at infinity and
PH log CDF for Exponential/Weibull (1534 passed / 8 skipped / 3 failed in
float64). Their value checks now use ordinary accuracy; derivative checks are
unchanged, and the cure-fraction case now adds log-parameter gradient/Hessian
checks against hand derivatives. Full touched infinity/PH files pass 58 tests
per dtype after these updates. Those are implementer tests, not spec tests.

## Reviewer checkout repair

The maintainer explicitly authorized discarding this throwaway checkout's stale
files. Outside the reviewer's restricted process, checkout --detach --force
c4023cf392a8fb71feb5187af15e396514c244b9 and clean -fd limited to tests/spec
restored review/code-fastlane. Read-only checks confirmed clean status, tester
SOURCE_COMMIT a0e7f49 and byte-identical Slice B fixture. This one-off repair
did not change reviewer roles/settings/prompts or spec-test contents. Normal
skill setup checks apply to this new step's review.

## Gate, commits and remaining work

Full `scripts/check.sh` passes on the final native implementation, with the
entire baseline and all five committed features enabled:

| Stage | Float64 | Float32 |
| --- | --- | --- |
| Implementer tests | 1537 passed, 8 skipped | 1536 passed, 9 skipped |
| Source doctests | 4 passed | 4 passed |
| Spec tests | 4899 passed, 47 skipped | 4810 passed, 126 skipped |

Guard passed; Ruff reports 307 files formatted and lint passed; mypy reports
no issues in 25 source files. Both spec runs have the two existing NumPy PH
reference warnings. The gate built and supplied its wheel and ended with
`check: all gates passed`.

`scripts/check.sh --fast` also passes with the refreshed inventory:

| Stage | Float64 | Float32 |
| --- | --- | --- |
| Implementer tests | 1411 passed, 8 skipped | 1410 passed, 9 skipped |
| Source doctests | 4 passed | 4 passed |
| Spec tests | 4826 passed, 47 skipped | 4759 passed, 104 skipped |

Static checks pass again; each spec run has the same two existing warnings.
Environment unchanged: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, CPU.
R and docs builds were not run; no R fixtures were regenerated.

The required inventory refresh completed with 201 slow tests. It adds 36 and
removes 357 generated lines after measuring the full suites. This
puts the combined step over the usual ~300-line target. The maintainer
explicitly approved including the generated refresh in this same step.
Implementation/test changes remain about 170 lines. Full and fast gates pass;
independent review follows this committed handoff.

Starting/previous step final: c4023cf392a8fb71feb5187af15e396514c244b9.
Original branch/worktree base: 3852184bb0e6f0a0a9b746ad8b92a95f675b23d0.
No new deviations. Prior D-SPEC62-63.1-001's missing-fixture failure was
eliminated by the maintainer's tester resync; formal disposition belongs to the
maintainer. D-SUMMARY.READINESS-001 remains closed as previously authorized.
No further implementation substeps planned here. Follow-up docs topic remains
in PLAN.md; maintainer integration is separate.

ROADMAP status: Revision-63 single-branch log1mexp implemented with native
highest-accuracy expm1; full/fast both-dtype gates and focused derivative suites pass.
Generated-inventory scope approved; commit and independent review pending.
Numerical-behaviour docs remain a follow-up.
