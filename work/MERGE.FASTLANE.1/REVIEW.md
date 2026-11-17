# MERGE.FASTLANE.1 — Resolve the generated-inventory conflict

Independent review: round 1/3

## Read first

1. PLAN.md and the two merge parents: cecdbcd (fastlane) and 99aff16 (main).
2. tests/slow_tests.txt: regenerated timings for the combined tree.
3. Incoming src/survjax/likelihood.py and
   tests/unit/test_ph_likelihood_shared_survival.py; implementation evidence is
   in the incoming PH.PERF.2/PH.PERF.3 records. These changes merged cleanly.
4. Prior SPEC62-63.1, SPEC63.LOG1MEXP.1 and SPEC63.NUMERICS-DOCS.1 records:
   the reviewed metadata, log1mexp and documentation changes are preserved.

## Scope and invariants

The maintainer started the main-to-fastlane merge and authorized resolution,
validation and final local-main integration. There is no next implementation
feature. Only tests/slow_tests.txt conflicted; all other incoming changes
were already staged by Git. We preserve those changes without rewriting
the incoming AIPTW tutorial, PH optimization or their historical records.
Incoming pyproject.toml changes only extend the existing notebook Ruff
exclusions; no dependency is added by the merge.

The conflict contained different generated timing runs (201 versus 937 slow
tests). Their timings cannot be honestly hand-merged. The incoming generated
snapshot was used temporarily, then scripts/update_slow_tests.sh regenerated
the file against the combined source in both dtypes with all five enabled
features. The final generated inventory has 133 tests, 421s of 813s summed
test time. This is local measured classification, not a performance claim.
The measurement command used PYTEST_WORKERS=6 and the shared Git-directory
JAX cache with JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0. The script ignores
pytest exit status while collecting timings; its completion alone is not
claimed as a passing gate. The full gate below provides that check.

No source conflict was resolved by choosing away one side. Explicit inspection
confirmed strong boolean-mask normalization and native HIGHEST log1mexp remain.
The PH optimization uses the existing concrete wrapper/static-metadata guard;
the focused matrix checks numerical agreement and fallback behavior. No
protected file, public API, tolerance or completed-feature selection changes
are introduced by this resolution. The generated conflict file is the only
existing file manually replaced; its final content comes from the script.

## Validation

No unmerged paths remain after staging tests/slow_tests.txt. git diff --check
passes. In each dtype mode (JAX_ENABLE_X64=1 and 0), ran in full:

`uv run --group test pytest -n 6 -q tests/unit/test_ph_likelihood_shared_survival.py tests/unit/test_data_metadata.py tests/unit/test_log1mexp.py tests/unit/test_log_cdf_tails.py`

Result: 88 passed per dtype. This covers incoming PH shared-survival values,
joint derivatives, validation/trace reuse and fallbacks, plus retained metadata
compilation reuse and log-CDF value/derivative accuracy (spec 3.2, 7.2–7.3,
12.1). The incoming benchmark results are historical; no new benchmark is run.

Full `scripts/check.sh` passes for the combined tree with the entire baseline
and all five committed features:

| Stage | Float64 | Float32 |
| --- | --- | --- |
| Implementer tests | 1550 passed, 8 skipped | 1549 passed, 9 skipped |
| Source doctests | 4 passed | 4 passed |
| Spec tests | 4900 passed, 46 skipped | 4812 passed, 124 skipped |

Guard passed; Ruff reports 332 files formatted and lint passed; mypy reports
no issues in 25 source files. Both spec runs emitted only the two existing
NumPy PH reference warnings. The gate built its wheel and ended with
`check: all gates passed`. Inventory validation also confirms exactly 133
sorted unique entries, matching its header, with no conflict markers.
Docs builds, R fixture regeneration, remote publication and additional
features are not part of this conflict repair.

## Integration handoff

Step starting / previous step final:
cecdbcd8b154281a0f323cddd1f1aa83f56da8f2. Incoming main parent:
99aff16ea9d73447f0d7435cf6e2cbfdc926ac37. Original worktree base:
3852184bb0e6f0a0a9b746ad8b92a95f675b23d0. No new deviations or implementation
substeps. Merge commit: f53cdc6b35bf18486e5eb07e683bdf7d9129fdb0.
Independent round 1 returned `merge`, with no findings, in
`/Users/wamster3/dev/survjax/review/slices/MERGE.FASTLANE.1.md`.
The reviewer independently passed both fast and full gates and the focused
88-test matrix in each dtype. Its full spec counts were 4899 passed / 47
skipped (float64) and 4810 passed / 126 skipped (float32); implementation
counts matched ours. The small skip-count difference was not investigated;
neither run had failures. No fix-up implementation commits were needed.

Opened Diffview for the saved starting commit through the reviewed merge
commit in pane %26 (window @8). Final local-main integration uses a checked
fast-forward only. Do not rebase, push, tag or remove this worktree.

ROADMAP status: Fastlane implementation and docs are complete; generated merge
conflict resolved; focused checks and full both-dtype gate pass; independent
review says merge. Ready for the authorized local-main fast-forward.
