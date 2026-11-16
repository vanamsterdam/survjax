# SPEC62-63.1 — Revision 62–63 compatibility audit

Independent review: round 2/3

## Read first

1. PLAN.md and the b3830a2..9c052c2 spec diff, including revision notes 62–63.
2. src/survjax/_data_validation.py: explicit bool normalization in indicator.
3. tests/unit/test_data_metadata.py: eager/traced compilation reuse regression
   for event and bounds indicators.
4. DEVIATIONS.md: authorized closure of D-SUMMARY.READINESS-001.

## Change and invariants

Revision 62 argument-form metadata is already implemented by B4.3 and enabled.
The replacement test exposed an extra JIT specialization: in JAX 0.11.2,
comparing a traced boolean indicator with integer 1 produces a weakly typed
boolean result. Eager flag normalization produces a strong bool. Explicit bool
conversion makes the normalized exact masks agree and permits reuse.

No field values, issue bits, constructor signatures, static-flag logic, public
exports or dependencies change. The indicator helper serves both constructors.
Masks stay boolean arrays; their shape and values are unchanged. Existing
numerical code retains the two-branch log1mexp implementation, permitted by
revision 63. No spec/spec-test files or roadmap files were edited.

## Requirements and focused verification

The complete files below ran with -n 6, without a slow-test filter, separately
under JAX_ENABLE_X64=1 and 0, with all five committed features selected:

- tests/unit/test_data_metadata.py and test_data_tracing.py: spec 3.2/6.2/7.2,
  static flags, normalized masks, tracing and compilation reuse.
- tests/spec/test_observation_traced_metadata_forms.py: revision 62 form rules.
- tests/spec/test_a4_log_cdf_upper_tail.py: generic log F values and derivatives.
- tests/spec/test_a5_log_cdf_tails.py: built-in log F values and derivatives,
  including underflowed H and upper tails (revision 63).
- tests/spec/test_a11_ph_direct_methods.py: retained PH direct-method behavior.
- tests/spec/test_observation_censoring_numerics.py: upper-tail left-censored
  rows, wide intervals, and first/second parameter derivatives (7.2/12.1).

Initial run before the bool fix: float64 234 passed / 1 failed; float32
227 passed / 3 skipped / 1 failed. Both failed only the new eager/traced
metadata compilation-reuse test. After the fix plus regression/tracing tests:
float64 271 passed; float32 264 passed / 3 skipped. Each had two existing
NumPy PH reference warnings (log zero and multiply zero by infinity).

## Separately verified RMST disposition

Maintainer confirmed main validation here, RMST separately. Against isolated,
clean committed rmst snapshot 73466bdef07c25afa9102052006bc61aead11976:

`JAX_ENABLE_X64=1 SPEC_FEATURES=family.lognormal,summary.rmst uv run --group test pytest -q tests/spec/test_family_lognormal_summaries.py::test_rmst_at_a_large_cutoff_approaches_the_mean_from_below`

Result: 1 passed. Repeated with JAX_ENABLE_X64=0: 1 passed. The independent
tester's de5d015 correction is present through sync 3852184/tester 30ada60.
D-SUMMARY.READINESS-001 is closed under explicit maintainer instruction;
the original historical record is untouched. D-SPEC62-63.1-001 separately
records the missing acceptance fixture introduced by the sync.
Main retains staged RMST; summary.rmst was not enabled here.

## Gate and handoff

Environment: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, CPU.
Completed feature selection is unchanged:
observation.censoring, observation.truncation, observation.traced_metadata,
family.lognormal, family.loglogistic. summary.rmst is tested only in the
separate snapshot, as the maintainer requested.

| Check | float64 | float32 |
| --- | --- | --- |
| Full implementer suite through gate | 1528 passed, 8 skipped, 1 failed | 1527 passed, 9 skipped, 1 failed |
| Full spec suite, direct invocation | 4798 passed, 47 skipped, 78 setup errors | 4745 passed, 126 skipped, 42 setup errors |
| Source doctests | 4 passed | 4 passed |
| Spec wheel tests after supplying built wheel | 5 passed | 5 passed |
| Pre-review fast gate implementer suite | 1220 passed, 8 skipped, 1 failed | not reached |

`scripts/check.sh` failed in its float64 implementer suite on
test_slice_b_payloads_load_and_acceptance_copy_is_identical, FileNotFoundError
for tests/spec/fixtures/slice_b_likelihood.json. To complete both-dtype
coverage, `SURVJAX_X64_MODES=0 scripts/check.sh` was run separately and failed
on that same assertion. The sync deleted the fixture; the independent tests
and README still require it. See D-SPEC62-63.1-001. No failed assertion was
edited or bypassed. Both standard invocations exited before doctests/spec
stages, so these stages were run directly without marker filtering.

Direct spec command: `JAX_ENABLE_X64=1 SPEC_FEATURES=observation.censoring,observation.truncation,observation.traced_metadata,family.lognormal,family.loglogistic uv run --group test pytest tests/spec -q -n 6`, repeated with 0.
Each had two existing PH NumPy reference warnings. Setup errors comprise
73 (float64) / 37 (float32) missing-fixture errors and five wheel setup errors
per dtype: the direct invocation initially omitted SURVJAX_WHEEL. Built the
current wheel with `uv build --wheel -q -o /private/tmp/survjax-spec62-63-wheel`
and reran all of test_a9_wheel_smoke.py in both dtypes with SURVJAX_WHEEL set
to that wheel; all five passed in each mode. Thus the only unresolved test
cause is the deleted acceptance fixture; the full gate has NOT passed.

`scripts/check.sh --fast` was attempted and repeated to capture final output;
both stop in the float64 implementer suite on the missing fixture. Ruff format
reports 304 files formatted; Ruff lint passes; mypy reports no issues in 25
source files. `git diff --check` passes. Doctest command was
`JAX_ENABLE_X64=1 uv run --group test pytest src --doctest-modules -q`, repeated
with 0. Both built-wheel implementation installation tests ran in the full
implementation suites. No R fixtures were regenerated; R and docs builds were
not run for this code step. New regression durations are 0.64s and 0.22s,
so no slow-test inventory refresh is required.

Starting/previous checkpoint and worktree base:
3852184bb0e6f0a0a9b746ad8b92a95f675b23d0. Implementation commit reviewed:
aa5629d0a1688d724dcd14c8ab6325121c57f9b5. This final record-only commit
adds the terminal review outcome; no source or test changes follow review.
No remaining implementation substeps. Tester must restore/resync the deleted
fixture before the full gate can pass; maintainer integration remains.
RMST/mean work remains on its separate branch.

## Independent review outcome

Round 1/3 reviewed aa5629d0a1688d724dcd14c8ab6325121c57f9b5:
**VERDICT: maintainer**. Report:
[SPEC62-63.1](../../../review/slices/SPEC62-63.1.md).
No blockers, should-fix items or nits. S-1 agrees with open
D-SPEC62-63.1-001: tester must restore/resync the missing acceptance fixture.
No fix-up or further review round is appropriate for that maintainer question.

The reviewer independently reproduced the fast gate result (1220 passed /
8 skipped / 1 missing-fixture failure) and reports 36 passing metadata/tracing
tests per dtype. Its listed command did not set SPEC_FEATURES, so its
replacement feature tests are not independently established as selected by
that command; our focused full-file runs above explicitly enabled all five
features. The reviewer did not repeat full/numerical/RMST-snapshot checks or
re-read the full spec diff; mutation-check setup was permission-blocked.
Clarification: our pre-fix failure evidence is the independent spec test
test_traced_and_eager_data_with_equal_flags_share_one_compilation, not a
pre-fix run of the newly added unit regression. No mutation result is claimed.

Optional workspace diff hook succeeded: pane %112, window @32, displaying
3852184..aa5629d. Maintainer must restore/resync the fixture and rerun the full
both-dtype gate. No roadmap edit, merge, push or publication was performed.

## Restored-fixture validation (supersedes the earlier failed gate)

Resumption checkpoint: 29a91a79094999d7782f1c3346cf6ee141f25ce1, the
maintainer's merge of current main 69b796fa9192fb382bfd38bcbd80e3a67078fa0d.
Tester a0e7f49, synced in 28b54e2, restores the Slice B acceptance fixture and
the comprehensive test_observation_traced_metadata.py suite. A byte comparison
of the two Slice B fixture copies passed. No source or test changes are needed.
The original Step starting commit remains 3852184; previous handoff commit was
f73d79e. This follow-up changes only this step's records.

Full `scripts/check.sh` passed, without dtype/feature/slow-test filters:

| Stage | Float64 | Float32 |
| --- | --- | --- |
| Implementer tests | 1529 passed, 8 skipped | 1528 passed, 9 skipped |
| Source doctests | 4 passed | 4 passed |
| Spec tests | 4899 passed, 47 skipped | 4810 passed, 126 skipped |

Guard passed on fastlane. Ruff: 304 files formatted; lint passed. Mypy: no
issues in 25 source files. Both spec runs emitted only the two existing NumPy
PH reference warnings. The gate built and supplied its wheel and ended with
`check: all gates passed` for the entire baseline plus observation.censoring,
observation.truncation, observation.traced_metadata, family.lognormal and
family.loglogistic. Summary activation is unchanged; the separate RMST evidence
above remains the evidence for D-SUMMARY.READINESS-001.

The cause of D-SPEC62-63.1-001/reviewer S-1 no longer reproduces. Its formal
disposition remains with the maintainer. Pre-review `scripts/check.sh --fast`
also passes: implementer suites 1221 passed / 8 skipped (float64) and
1223 passed / 6 skipped (float32); source doctests 4 passed per dtype; spec
suites 4689 passed / 47 skipped (float64) and 4631 passed / 95 skipped
(float32). Both spec runs have the same two existing warnings. No test or
source change was needed. Independent review round 2 result follows when
complete. Display the saved starting commit through the final reviewed commit
with the workspace diff hook.

ROADMAP status: Rev 62–63 compatibility and full both-dtype gate pass after
tester fixture restoration; boolean-mask compilation reuse fixed;
D-SUMMARY.READINESS-001 closed. Review follow-up pending; integration remains.
