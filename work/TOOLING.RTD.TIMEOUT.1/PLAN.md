# TOOLING.RTD.TIMEOUT.1 — Avoid duplicate hosted docs execution

Step starting commit: 00b3d1c6798ee12729018c359828ef2dfaf2b896
Worktree base: 54e3fa9f280f63bcd9c08865fc99e128cd69c370.
Branch: bayesian-use-case. Preceding step: merge main into this PR (00b3d1c).
Dependencies: current scripts/build_docs.sh and its isolation/certification tests.
Completed feature selection: unchanged.

RTD build 35961090 for this HEAD died after exactly 900 seconds. Its raw log
records dependency installation in 10 seconds and the unfinished command
scripts/build_docs.sh. The hosted job currently executes both modes but
publishes only html-x64. No per-page timing survives the killed command.

Tooling step, under 300 changed implementation lines: add --dtype
{both,x32,x64} to scripts/build_docs.sh, default both. Make certification and
printed output paths describe only the modes actually executed. Change
.readthedocs.yaml to request x64 only. Update DOCS-BUILD.md's CLI contract.
Add meaningful tests for selection, invalid input, accurate identity and
failed single-mode rebuild invalidation; preserve existing default both-mode
and worktree isolation checks. No docs/ source, dependencies, library API,
protected-path or spec changes. One new local tooling CLI option, no module.

Spec 10.5: retain the documented default command that executes examples afresh
in both applicable dtype processes and fails on warnings/errors. Hosting can
select its published mode without weakening the default verification command.
No spec deviation needed; this changes publication workload, not that gate.

Validate touched tooling tests, CLI help/errors and shell syntax, both-mode
strict build via scripts/build_docs.sh, selected-mode strict build via
scripts/build_docs.sh --dtype x64, rendered output and mode identity. Run the
fast gate, commit handoff, use independent-review, push to existing PR and
observe RTD if possible. Do not merge. A local run cannot establish RTD's time;
record the hosted outcome rather than claim the 15-minute issue solved unseen.

## Docs to-do

Root DOCS-BUILD.md documents this tooling contract within the same step.
No tutorial-content changes. The diff hook's nvim.log is unrelated and will be
preserved outside the checkout before review.
