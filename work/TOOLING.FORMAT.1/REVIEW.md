# TOOLING.FORMAT.1 — handoff

Independent review: skipped (0/3), per the maintainer's instruction to test
only make format for this follow-up and merge if it passes.

## Read first

1. Makefile format target: both Ruff write commands exclude spec, tests/spec
   and legacy plans. The CLI exclusion includes spec explicitly, preserving
   the pyproject exclusion when overriding this setting.
2. [PLAN.md](PLAN.md): starting commit, scope and validation agreement.

## Invariants and evidence

No library code, public API, dependencies, protected files or existing step
records changed. Only the format target and this step's records changed.

Executed make format in the separate pr11-format worktree:

- ruff format: 262 files left unchanged.
- ruff check --fix: All checks passed!
- SHA-256 snapshots before/after: all 183 files under spec, tests/spec and
  plans unchanged. git diff showed only the intended Makefile change.

No full suite, fast gate, docs build or independent review run for this
follow-up, as requested. PR #11's remote Plan, Hooks and Linux/macOS CPU
jobs (both dtypes) passed at 2f1c510dbdbdfd5ad1412a682331ae3372a7e15f.

## Integration

Preserve Juan Orduz's two original commits and author attribution; add this
fix as a separate commit. Local main advanced to
a1aad2dcdcd5e698d4931b9d58534923fd39fa9a with unrelated documentation changes
while this work proceeded. Merge without rebasing the published PR commits.
No further sub-steps. Hook-migration guidance and protected instruction/spec
wording remain outside this focused formatting change.
