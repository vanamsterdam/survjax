# BAYES.CENSOR20.9 — Fixed-limit trial handoff

Independent review: round 2/3

## Read first

1. PLAN.md and DEVIATIONS.md: explicit maintainer request for 20.0 before
   merging PR #12, overriding the calibrated-limit design note.
2. model.py make_trial and the notebook trial cell. Both use 20.0; the runner
   still calls the shared generator, whose implementation is unchanged.
3. tests/unit/check_bayesian_run.py and regenerated results/metadata.json.
4. Notebook saved output: c=20.0, 157/300 censored, refreshed summaries/plots.

## Invariants and changes

Only the Bayesian censoring limit changes. Keep the event-time law, shape,
scale, HR, seeds, U/C split, arm order, precision, priors, NUTS configuration
and exact/PH/RMST calculations. The notebook and runner retain bit-for-bit
trial equality. The regression assertion independently fixes the requested
limit to 20.0. The unused tmp_path fixture argument is removed (prior nit).
The notebook's horizon endpoint is now 13.5 (observed 90th percentile rounded
down to half a year), matching the regenerated posterior.npz grid.

The metadata and NPZ were generated together by the unchanged runner, with
four chains, 1000 warmup and 1000 retained draws. Notebook outputs were all
executed afresh, including plots and execution metadata; its NUTS key remains
different from the runner as documented. No results were manually fabricated.
No library code/API, dependencies, protected paths or spec files changed.

## Validation

- `scripts/check.sh --fast`: exit 0, all gates passed with the unchanged
  committed feature selection in both dtypes. Ruff: 310 files formatted,
  all checks passed; mypy: 25 source files clean. Implementation x64:
  1219 passed/8 skipped, x32: 1221 passed/6 skipped; doctests: 4 passed
  per dtype. Final x32 spec run: 4597 passed/93 skipped, with the two existing
  NumPy log-zero warnings. Tier 2 was not rerun after this change; docs build
  belongs to the pending docs step. Notebook was executed in full instead.
- `uv run --group examples python -m examples.bayesian_rmst.run
  /tmp/survjax-censor20-run`: exit 0, all acceptance assertions passed.
  Saved JSON/NPZ copied to examples/bayesian_rmst/results. c=20.0, censored=157,
  divergences=0; maximum R-hat about 1.003825, minimum ESS about 1811.93;
  truth inside the HR and delta-RMST intervals. RMST routes differ by at most
  2.842170943040401e-14 years. Truth parameters/RMST are unchanged.
- Fresh nbclient execution from a TemporaryDirectory outside checkout: exit 0,
  all outputs saved back to notebook. Displays c=20.0 and 157 censored, zero
  divergences, and 1.33e-14 scalar RMST route difference. IPython used a temp
  directory for unwritable ~/.ipython; kernel cleanup logged sandbox psutil
  sysctl PermissionError after execution. These did not fail execution.
- `uv run --group docs --group examples --group test pytest -q
  tests/unit/check_bayesian_run.py`: 3 passed in 22.15s. Demonstrates trial
  equality and 20.0 regression, outside-checkout execution with examples
  unavailable, and all saved posterior arrays reproduced bit for bit.
- Ruff format/check on touched Python files: 2 files unchanged, all checks
  passed. git diff --check clean. Notebook JSON parsed and saved output
  inspected; regenerated metadata provenance now records survjax 0.0.1a3.

## Remaining sub-step / limits

This is the code/results step. Docs sync remains necessary before merging:
the page's 19.2 years, 164 censored, 55% expectation, 13-year horizon and ESS
above 2000 claims are stale; the README/page calibrated-limit wording also
needs updating. See PLAN.md Docs to-do. No docs/ changes in this code step.
The explicit fixed-limit exception is logged open for maintainer resolution;
no spec edits are inferred from the runtime-value request.
The maintainer withdrew the merge instruction; PR #12 was confirmed OPEN with
no mergedAt/mergeCommit. Push this reviewed step to the existing PR, never
merge it. Generated nvim.log preserved at /tmp/survjax-before-censor20-nvim.log.

## Round 1 response

Reviewed commit: d88d3aaa659bf98481b68908ec4b4cc298a76c12.
Verdict: fix-first. Finding IDs: F-1, N-1, S-1.

- F-1: removed the three stderr outputs from the notebook setup cell (tqdm
  widgets warning and Matplotlib cache messages with local paths). The cell
  again has no saved outputs. All calculated outputs/plots and executable
  source stay unchanged. Scanned every saved output: no /Users/ or
  /var/folders/ paths remain. No dependency added for a warning-only display.
- N-1: directly checked np.percentile(saved time, 90), rounded down to half
  a year, against saved tau_grid[-1]; both yield 13.5. This validates the
  horizon claim without changing it.
- S-1: maintainer owns resolution of the logged explicit fixed-limit request.
  No spec changes or deviation resolution made by the implementer.
- Reviewer passed the exact trial/20.0 check and posterior reproduction;
  notebook execution could not start because its sandbox blocks local ports.
  Its fast gate was incomplete at review write time, with no observed failures.
- Output-only fix validation: notebook JSON parses, setup outputs empty, no
  local paths in saved outputs, horizon assertion passes, git diff --check
  clean. No numerical/source changes, so no NUTS/test repeat for this cleanup.
