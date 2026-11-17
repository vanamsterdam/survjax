# TOOLING.RTD.TIMEOUT.1 — Hosted docs workload handoff

Independent review: round 1/3

## Read first

1. PLAN.md; failed RTD build 35961090 (commit 00b3d1c).
2. scripts/build_docs.sh argument selection, identity and final printed paths.
3. .readthedocs.yaml publication command and DOCS-BUILD.md interface contract.
4. Added selection/failure tests in tests/unit/test_docs_build_script.py and
   tests/tooling/test_build_docs.py, alongside unchanged default/isolation tests.

## Evidence and invariants

RTD API reports duration=900, success=false. Raw log at
https://app.readthedocs.org/api/v2/build/35961090.txt shows locked dependency
installation taking 10 seconds and scripts/build_docs.sh unfinished. The
killed command has no retained Sphinx output, so no particular page is blamed.
The prior command executes two sites but publishes only html-x64.

--dtype both remains the default. It still executes each notebook afresh in
separate x32/x64 processes, uses strict Sphinx warnings/errors, isolates the
worktree environment and caches, and certifies success only after all selected
builds and checkout-stability checks pass. RTD explicitly selects x64 and
publishes it only on successful completion. This preserves spec 10.5's full
verification entry point without requiring the hosting job to duplicate it.
No notebooks are skipped or cached as a replacement for fresh execution.

A single-mode certificate names only that mode. Old other-mode HTML is neither
deleted nor certified, and its path is not printed. Failed single-mode runs
invalidate previous pair certificates. Invalid dtype values fail before setup.
No library API/dependencies, protected files or docs/ sources changed.

## Validation

- JAX_ENABLE_X64=1 and =0, separately: `uv run --group test pytest -n 6
  tests/unit/test_docs_build_script.py tests/tooling/test_build_docs.py`:
  24 passed in 7.11s and 24 passed in 6.29s. Tests cover actual shell execution,
  modes/paths, preserved default behavior, stale-output certification, bad
  arguments, failed rebuilds and worktree/platform isolation.
- `scripts/build_docs.sh`: strict executed x32 and x64 sites passed, exit 0,
  identity lists x32 x64 and ends state: success; elapsed 201 seconds locally.
- `scripts/build_docs.sh --dtype x64 --output-dir
  /tmp/survjax-rtd-x64-validation`: strict executed site passed, exit 0,
  identity lists x64 only and ends state: success; elapsed 96 seconds locally.
  Fresh output/cache directory; existing environment/MNIST input cache reused.
  No html-x32 directory produced. Local timings are observations, not a claim
  about RTD hardware or an isolated performance benchmark.
- Parsed identity records and checked index, API, AIPTW and Bayesian tutorial
  HTML exist for every certified mode. This changes build plumbing, not layout;
  no screenshot comparison needed. Standard kernel TCP notices appeared.
- `scripts/check.sh --fast`: exit 0, all gates passed with unchanged committed
  feature selection. Ruff: 344 files formatted, all checks passed; mypy: 25
  source files clean. Implementation x64: 1453 passed/8 skipped; x32: 1452
  passed/9 skipped. Doctests: 4 passed per mode. Final x32 spec stage:
  4802 passed/105 skipped, two existing NumPy log-zero warnings. Tier 2 not run
  for this tooling change; both touched test files ran without slow filtering.
- `bash -n scripts/build_docs.sh`, CLI --help, Ruff on touched tests and
  git diff --check: passed.

## Remaining limits and integration

Push reviewed change to PR #12 and observe the next hosted build; the local
builds cannot establish completion within RTD's 15-minute ceiling. Do not
merge. No spec deviation or content follow-up required. Original worktree base
and preceding commit are in PLAN.md. Generated nvim.log was preserved at
/tmp/survjax-before-rtd-timeout-nvim.log.
