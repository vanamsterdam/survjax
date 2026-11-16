# TOOLING.FORMAT.1 — Protect read-only inputs from make format

Step starting commit: 2f1c510dbdbdfd5ad1412a682331ae3372a7e15f
Feature: contributor formatting command; no feature-selection changes.
Branch/worktree base: 2f1c510dbdbdfd5ad1412a682331ae3372a7e15f (PR #11).
Previous step: [TOOLING.MAKE-PREK](../TOOLING.MAKE-PREK/PLAN.md), ending at the same commit.
PR base on main: 4d97c943a5570164adf0d3ef83d6748a39e15c54.
Dependency: PR #11's Makefile and locked Ruff environment.

## Plan

Change only the Makefile format target: exclude spec, tests/spec and legacy
plans from both formatting and lint fixes. Keep the existing spec exclusion
explicit because command-line settings override the configured value.
Public API: none. Spec coverage: 10.1 (quality tools); repository working
agreement (read-only independent tests, spec and legacy records).
No new tests: this is a reversible command-scope correction.
Run only make format, checking that excluded files retain identical contents
and the target produces no unrelated tracked changes. The maintainer requested
this focused validation instead of rerunning the full suite; PR #11's existing
remote hooks job and four CPU jobs passed at the starting commit.
Independent review is omitted for this small follow-up under the maintainer's
instruction to test only the format step and then merge if it passes.

## Docs to-do

None. No further sub-steps.
