# SPEC62-63.1 — Deviation disposition

## D-SUMMARY.READINESS-001 — Closed by maintainer instruction

- Original step: SUMMARY.READINESS; spec 8.1, 8.8, 12.1.
- Original record: [SUMMARY.READINESS](../../../dev-rmst/work/SUMMARY.READINESS/DEVIATIONS.md).
- Issue: the float32 log-normal RMST mean upper-bound assertion allowed only
  float64-scale rounding, rejecting the nearest representable float32 mean.
- Resolution supplied by the independent tester: de5d015, included in tester
  30ada60 and main sync 3852184. The bound allows rounding in the active dtype;
  the independent RMST accuracy assertion remains.
- Verification: test_family_lognormal_summaries.py::
  test_rmst_at_a_large_cutoff_approaches_the_mean_from_below passed once in
  each dtype against committed RMST snapshot
  73466bdef07c25afa9102052006bc61aead11976, with
  SPEC_FEATURES=family.lognormal,summary.rmst. The snapshot is isolated from
  ongoing uncommitted RMST work. This main-based branch retains staged RMST.
- Status: closed, explicitly authorized by the maintainer's current task.
  This disposition supersedes the original open status; its historical file
  is unchanged. No new spec deviation was opened.

## D-SPEC62-63.1-001 — Missing Slice B acceptance fixture

- Step: SPEC62-63.1.
- Spec section: 10.4, 12.1–12.2 (committed independent R likelihood evidence).
- Spec says: the Slice B likelihood fixture is committed and tests compare
  the applicable ten censoring/truncation cells against fixed flexsurv output.
- Observed: main sync 3852184 (tester 30ada60) deleted
  tests/spec/fixtures/slice_b_likelihood.json. The acceptance tests and their
  fixture README still require that path. The implementer fixture remains at
  tests/fixtures/slice_b/slice_b_likelihood.json with its manifest.
- Evidence: full gate float64 implementer suite failed
  tests/reference/test_slice_b_r.py::test_slice_b_payloads_load_and_acceptance_copy_is_identical
  with FileNotFoundError at the missing acceptance path; 1528 passed / 8 skipped.
- Proposal: independent tester restores the fixture and maintainer resyncs it.
  No fixture is regenerated, copied into tests/spec, skipped or xfailed here;
  the byte-identity assertion remains unchanged. Remaining gate stages run
  directly to expose other failures without bypassing the missing fixture.
- Status: open; tester/maintainer action requested during this task.

### Restored-fixture verification

The maintainer supplied tester a0e7f49 through main 69b796f and merged it into
fastlane at 29a91a7. The restored acceptance fixture is byte-identical to the
implementer fixture. Full scripts/check.sh now passes in both dtypes with all
five committed features, including the fixture integrity and R reference tests.
The reported failure no longer reproduces; formal disposition remains with the
maintainer. No protected path was edited by this step.
