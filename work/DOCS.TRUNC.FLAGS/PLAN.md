# DOCS.TRUNC.FLAGS — Flag-based censored and truncated observations

Step starting commit: cd383be231739ee8851d2dac540b00289128f622
Feature: documentation for observation.censoring and observation.truncation.
Worktree base: cd383be231739ee8851d2dac540b00289128f622; no additional dependencies.
Prior documentation: [B6.1](../B6.1/PLAN.md).
Completed feature selection: the existing selection in scripts/spec-features.txt.

Update docs/likelihood.md with the flag constructor equivalent to its existing
bounds example, and check its likelihood against the same independent expected
values. No public API changes or standalone tests. Covers spec sections 3 and 6.
Run scripts/build_docs.sh and scripts/check.sh --fast. Docs to-do: none.
